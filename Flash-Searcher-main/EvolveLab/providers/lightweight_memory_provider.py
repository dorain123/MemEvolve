"""
Lightweight Memory Provider

A streamlined, efficient dual-memory system:
1. Short-term Memory: Task-level key information, constraints, and conditions
2. Long-term Memory: Compact JSON storage with 30 strategic + 30 operational memories

Core Features:
- Short-term memory: Provided every 3 steps during execution
- Long-term memory: Only provided at BEGIN phase (top 5 most relevant)
- Minimal overhead: Simple JSON storage, no vector indices
- LLM-driven selection: Intelligent matching and synthesis
"""

import json
import os
import re
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional

from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import (
    MemoryRequest,
    MemoryResponse,
    MemoryItem,
    MemoryItemType,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)


# =========================================================================
# Cold-Start Long-term Memories
# =========================================================================
COLDSTART_STRATEGIC_MEMORIES = [
    {
        "content": "Execute directly rather than over-planning. Prioritize immediate action and concrete steps over extensive theoretical analysis.",
        "tags": ["execution", "planning"],
    },
    {
        "content": "Interpret task instructions literally and precisely. Pay close attention to output format requirements (units, separators, capitalization).",
        "tags": ["instruction", "format"],
    },
    {
        "content": "For numerical tasks, identify all units explicitly and track unit conversions through each step of calculation.",
        "tags": ["computation", "units"],
    },
    {
        "content": "When tasks present contradictory instructions that test the agent's internal logic or ability to follow specific directives, prioritize direct compliance with the explicit instruction over meta-analysis of the contradiction.",
        "tags": ["instruction", "contradiction", "compliance"],
    },
    {
        "content": "If there is no direct or first-hand authoritative evidence, other sources or the most likely answer analyzed can also be provided when reporting the answer, as long as it is clearly labeled.",
        "tags": ["evidence", "reporting", "transparency"],
    },
]

COLDSTART_OPERATIONAL_MEMORIES = [
    {
        "content": "When crawling web pages: Use specific queries for exact information, extract numbers with full context (units, date, conditions).",
        "tags": ["web_search", "extraction"],
    },
    {
        "content": "For multi-source information tasks: Navigate to exact source, extract verbatim values, write formula with units, apply formatting only at end.",
        "tags": ["procedure", "information_retrieval"],
    },
]


class LightweightMemoryProvider(BaseMemoryProvider):
    """
    Lightweight Memory System: Short-term (task-level) + Long-term (strategic/operational)
    
    Architecture:
    1. Short-term Memory: In-memory list of key facts/constraints for current task
    2. Long-term Memory: JSON file with 30 strategic + 30 operational memories
    3. Frequency Control: Short-term every 3 steps, long-term only at BEGIN
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(memory_type=MemoryType.LIGHTWEIGHT_MEMORY, config=config or {})
        
        # Logger configuration
        self.logger = logging.getLogger(f"{__name__}.Lightweight")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [Lightweight] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # LLM model
        self.model = self.config.get("model", None)
        
        # Storage paths
        self.storage_dir = self.config.get("storage_dir", "storage/lightweight_memory")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.longterm_memory_path = self.config.get(
            "longterm_memory_path",
            os.path.join(self.storage_dir, "longterm_memory.json")
        )
        
        # Configuration parameters
        self.max_strategic_memories = int(self.config.get("max_strategic_memories", 30))
        self.max_operational_memories = int(self.config.get("max_operational_memories", 30))
        self.max_shortterm_items = int(self.config.get("max_shortterm_items", 10))
        self.shortterm_provision_interval = int(self.config.get("shortterm_provision_interval", 3))
        self.top_k_longterm = int(self.config.get("top_k_longterm", 5))
        self.enable_longterm_provision = bool(self.config.get("enable_longterm_provision", False))
        self.longterm_min_confidence = float(self.config.get("longterm_min_confidence", 0.35))
        self.begin_memory_budget_chars = int(self.config.get("begin_memory_budget_chars", 1600))
        self.in_memory_budget_chars = int(self.config.get("in_memory_budget_chars", 900))
        self.max_failure_patterns = int(self.config.get("max_failure_patterns", 30))
        
        # Memory buffer configuration (allow expansion before pruning)
        self.memory_buffer_size = int(self.config.get("memory_buffer_size", 20))
        
        # Counter for generating unique IDs
        self._memory_id_counter = 0
        self._task_id_counter = 0
        
        # Long-term memory database
        self.longterm_db = None
        
        # Task-level snapshots: {query_hash: [memory_id1, memory_id2, ...]}
        # Preserves used_memory_ids per task to prevent race conditions in concurrent execution
        # Snapshot is created in BEGIN phase and cleaned up immediately after ingestion
        self._task_snapshots = {}
        
        # Short-term memory: task-level storage
        self.shortterm_memory = []  # List of key facts/constraints
        
        # Task context
        self.task_context = {
            "task_id": None,
            "query": None,
            "start_time": None,
            "current_step": 0,
            "last_shortterm_provision_step": -999,
            "longterm_provided": False,  # Only provide once at BEGIN
            "last_context": "",  # Store last step's context for delta calculation
            "agent_steps": [],  # Store step summaries: [{"step": N, "summary": "...", "timestamp": ...}, ...]
            "used_memory_ids": [],  # Track which long-term memories were used in this task
        }

    # =========================================================================
    # Core API Implementation
    # =========================================================================

    def initialize(self) -> bool:
        """Initialize the memory system: load long-term memories"""
        try:
            self.logger.info("=== Initializing Lightweight Memory System ===")
            
            # Load long-term memory database
            items_before = 0
            if os.path.exists(self.longterm_memory_path):
                try:
                    with open(self.longterm_memory_path, "r", encoding="utf-8") as f:
                        existing_db = json.load(f)
                        items_before = (
                            len(existing_db.get("strategic", [])) + 
                            len(existing_db.get("operational", [])) +
                            len(existing_db.get("failure_patterns", []))
                        )
                except Exception:
                    pass
            
            self.longterm_db = self._load_longterm_db()
            items_after = (
                len(self.longterm_db.get("strategic", [])) + 
                len(self.longterm_db.get("operational", [])) +
                len(self.longterm_db.get("failure_patterns", []))
            )
            self.logger.info(
                f"Long-term memory loaded: {len(self.longterm_db.get('strategic', []))} strategic, "
                f"{len(self.longterm_db.get('operational', []))} operational, "
                f"{len(self.longterm_db.get('failure_patterns', []))} failure patterns "
                f"(max: {self.max_strategic_memories}/{self.max_operational_memories}, "
                f"trigger pruning at: {self.max_strategic_memories + self.memory_buffer_size}/"
                f"{self.max_operational_memories + self.memory_buffer_size})"
            )
            
            # Save database if cold-start memories were injected
            if items_before == 0 and items_after > 0:
                self.logger.info("Saving cold-start memories to disk...")
                self._save_longterm_db()
                self.logger.info("Cold-start memories saved successfully")
            
            self.logger.info("=== Lightweight Memory Initialization Complete ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            return False

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Provide memory based on task phase and frequency control
        
        Flow:
        1. BEGIN phase: Provide long-term memory guidance (only once)
        2. IN phase: Provide short-term memory (every N steps)
        """
        if self.longterm_db is None:
            self.logger.warning("System not initialized, initializing now")
            if not self.initialize():
                raise RuntimeError("Failed to initialize Lightweight memory system")
        
        self.logger.info("=" * 80)
        self.logger.info("LIGHTWEIGHT MEMORY RETRIEVAL START")
        self.logger.info(f"Query: {request.query[:150]}...")
        self.logger.info(f"Phase: {request.status.value}")
        
        try:
            # ===== BEGIN Phase: Detect new task and reset =====
            if request.status == MemoryStatus.BEGIN:
                # Check if this is a new task (different query or no task initialized)
                if (self.task_context["query"] is None or 
                    self.task_context["query"] != request.query):
                    self.logger.info("🔄 New task detected, resetting task context")
                    self.reset_task_context(query=request.query)
            
            # Increment step counter
            self.task_context["current_step"] += 1
            current_step = self.task_context["current_step"]
            
            memories = []
            
            # ===== BEGIN Phase: Provide long-term memory (only once) =====
            if request.status == MemoryStatus.BEGIN:
                if not self.enable_longterm_provision:
                    self.logger.info("🚫 Long-term memory provision is disabled via config")
                elif not self.task_context["longterm_provided"]:
                    self.logger.info("BEGIN phase: Retrieving long-term memory...")
                    longterm_guidance = self._retrieve_longterm_memory(request)
                    if longterm_guidance:
                        memories.append(longterm_guidance)
                        self.task_context["longterm_provided"] = True
                        self.logger.info("Long-term memory provided")
                else:
                    self.logger.info("Long-term memory already provided, skipping")
            
            # ===== IN Phase: Auto-extract + Provide short-term memory =====
            elif request.status == MemoryStatus.IN:
                # Auto-extract key information from current step (every step)
                self._auto_extract_shortterm(request)
                
                # Provide accumulated short-term memory (every N steps)
                last_provision = self.task_context["last_shortterm_provision_step"]
                steps_since_last = current_step - last_provision
                
                if steps_since_last >= self.shortterm_provision_interval:
                    self.logger.info(f"Providing short-term memory (step {current_step})")
                    shortterm_guidance = self._retrieve_shortterm_memory(request)
                    if shortterm_guidance:
                        memories.append(shortterm_guidance)
                        self.task_context["last_shortterm_provision_step"] = current_step
                        self.logger.info("Short-term memory provided")
                else:
                    self.logger.info(
                        f"🚫 FREQUENCY CONTROL: Skipping short-term memory "
                        f"(steps since last: {steps_since_last} < {self.shortterm_provision_interval})"
                    )
            
            self.logger.info(f"LIGHTWEIGHT RETRIEVAL COMPLETE: {len(memories)} memories")
            self.logger.info("=" * 80)
            
            return MemoryResponse(
                memories=memories,
                memory_type=MemoryType.LIGHTWEIGHT_MEMORY,
                total_count=len(memories),
            )
            
        except Exception as e:
            self.logger.error(f"Memory retrieval error: {str(e)}", exc_info=True)
            return MemoryResponse(
                memories=[],
                memory_type=MemoryType.LIGHTWEIGHT_MEMORY,
                total_count=0,
            )

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """
        Absorb new memory from trajectory
        
        Flow:
        1. Extract strategic and operational memories from trajectory
        2. Update long-term memory (maintain max limits)
        3. Clear short-term memory for next task
        """
        if self.longterm_db is None:
            self.logger.warning("System not initialized, initializing now")
            if not self.initialize():
                return (False, "Failed to initialize memory system")
        
        self.logger.info("=" * 80)
        self.logger.info("LIGHTWEIGHT MEMORY INGESTION START")
        self.logger.info(f"Query: {trajectory_data.query[:150]}...")
        
        try:
            absorbed_items = []
            
            # Only extract from successful trajectories
            is_success = self._is_trajectory_success(trajectory_data)
            
            # Extract full_query from metadata for snapshot matching (if available)
            metadata = trajectory_data.metadata or {}
            full_query = metadata.get("full_query") or trajectory_data.query
            self._update_memory_feedback_counts(metadata.get("memory_feedback"))
            
            # Update success_count for used memories
            if is_success:
                self._update_memory_success_count(query=full_query)
            
            if is_success:
                self.logger.info("Trajectory successful, extracting memories...")
                
                # Extract strategic and operational memories
                extraction = self._extract_memories(trajectory_data)
                
                if extraction:
                    strategic_added = self._add_strategic_memories(
                        extraction.get("strategic", [])
                    )
                    operational_added = self._add_operational_memories(
                        extraction.get("operational", [])
                    )
                    
                    absorbed_items.extend([f"strategic:{i}" for i in range(strategic_added)])
                    absorbed_items.extend([f"operational:{i}" for i in range(operational_added)])
                    
                    self.logger.info(
                        f"Added {strategic_added} strategic, {operational_added} operational memories"
                    )
            else:
                self.logger.info("Trajectory not successful, extracting failure patterns...")
                failure_patterns = self._extract_failure_patterns(trajectory_data)
                failure_added = self._add_failure_patterns(failure_patterns or [])
                absorbed_items.extend([f"failure_pattern:{i}" for i in range(failure_added)])
                self.logger.info(f"Added {failure_added} failure pattern memories")
            
            # Save long-term database
            self._save_longterm_db()
            
            # Clean up task snapshot immediately after ingestion (using full_query for matching)
            if full_query:
                query_key = self._compute_signature(full_query)
                if query_key in self._task_snapshots:
                    del self._task_snapshots[query_key]
                    self.logger.info(
                        f"🗑️  Deleted task snapshot: hash={query_key[:16]}..., "
                        f"query_preview={full_query[:80]}..."
                    )
                else:
                    self.logger.warning(
                        f"⚠️  Snapshot to delete NOT found: hash={query_key[:16]}..."
                    )
            
            # Note: Do NOT reset task_context here! It may belong to another task already in progress.
            # Task context reset is handled in provide_memory(BEGIN) when a new task is detected.
            
            self.logger.info(f"LIGHTWEIGHT INGESTION COMPLETE: {len(absorbed_items)} items")
            self.logger.info("=" * 80)
            
            description = f"Lightweight memory absorbed {len(absorbed_items)} items"
            return (True, description)
            
        except Exception as e:
            self.logger.error(f"Memory ingestion error: {str(e)}", exc_info=True)
            return (False, f"Ingestion failed: {str(e)}")

    # =========================================================================
    # Long-term Memory Retrieval (BEGIN phase only)
    # =========================================================================

    def _retrieve_longterm_memory(self, request: MemoryRequest) -> Optional[MemoryItem]:
        """
        Retrieve long-term memory guidance for BEGIN phase
        
        Uses LLM to:
        1. Select top-K most relevant strategic/operational memories
        2. Synthesize into concise guidance
        """
        if not self.model:
            self.logger.warning("No model available for long-term memory retrieval")
            return None
        
        try:
            strategic_memories = self.longterm_db.get("strategic", [])
            operational_memories = self.longterm_db.get("operational", [])
            failure_patterns = self.longterm_db.get("failure_patterns", [])
            
            if not strategic_memories and not operational_memories and not failure_patterns:
                self.logger.info("No long-term memories available")
                return None
            
            # Build candidate list with scores
            candidates = []
            
            for memory_type, memory_list in [
                ("strategic", strategic_memories),
                ("operational", operational_memories),
                ("failure_pattern", failure_patterns),
            ]:
                for idx, mem in enumerate(memory_list):
                    if not isinstance(mem, dict):
                        mem = self._normalize_memory_record(mem, memory_type=memory_type)
                        memory_list[idx] = mem
                    confidence = self._memory_confidence(mem)
                    usage_count = mem.get("usage_count", 0)
                    harmful_count = mem.get("harmful_count", 0)
                    unsupported_count = mem.get("unsupported_count", 0)
                    harmful_rate = harmful_count / usage_count if usage_count else 0.0
                    
                    if confidence < self.longterm_min_confidence and usage_count == 0:
                        continue
                    if usage_count >= 2 and harmful_rate >= 0.5:
                        continue
                    if unsupported_count >= 2 and mem.get("success_count", 0) == 0:
                        continue
                    
                    candidates.append({
                        "id": f"{memory_type}_{idx}",
                        "type": memory_type,
                        "content": mem.get("content", ""),
                        "usage_count": usage_count,
                        "success_rate": self._calculate_success_rate(mem),
                        "harmful_rate": harmful_rate,
                        "confidence": confidence,
                        "applies_when": mem.get("applies_when", ""),
                        "do_not_apply_when": mem.get("do_not_apply_when", ""),
                        "provenance": mem.get("provenance", {}),
                    })
            
            if not candidates:
                self.logger.info("No long-term memories passed gating")
                return None
            
            # Use LLM to select and synthesize
            guidance, selected_memory_ids = self._select_and_synthesize_longterm(request, candidates)
            
            if not guidance or not guidance.strip():
                return None
            
            selected_candidates = [candidate for candidate in candidates if candidate["id"] in selected_memory_ids]
            selected_confidences = [candidate.get("confidence", 0.45) for candidate in selected_candidates]
            synthesized_confidence = min(selected_confidences) if selected_confidences else 0.45
            synthesized_provenance = {
                "source_url": None,
                "source_type": "memory_synthesis",
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "evidence_span": ", ".join(selected_memory_ids),
                "confidence": synthesized_confidence,
                "expires_at": None,
            }
            
            self._memory_id_counter += 1
            memory_item = MemoryItem(
                id=f"lightweight_longterm_{self._memory_id_counter}",
                content=guidance,
                metadata={
                    "source": "lightweight_longterm",
                    "phase": "begin",
                    "selected_memory_ids": selected_memory_ids,
                    "budget_chars": self.begin_memory_budget_chars,
                    "provenance": synthesized_provenance,
                    "selected_provenance": [
                        {
                            "memory_id": candidate["id"],
                            "provenance": candidate.get("provenance", {}),
                            "confidence": candidate.get("confidence"),
                            "applies_when": candidate.get("applies_when"),
                            "do_not_apply_when": candidate.get("do_not_apply_when"),
                        }
                        for candidate in selected_candidates
                    ],
                    "gating": {
                        "min_confidence": self.longterm_min_confidence,
                        "candidate_count": len(candidates),
                    },
                },
                type=MemoryItemType.TEXT,
            )
            
            return memory_item
            
        except Exception as e:
            self.logger.error(f"Long-term memory retrieval error: {str(e)}", exc_info=True)
            return None

    def _select_and_synthesize_longterm(
        self, 
        request: MemoryRequest, 
        candidates: List[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        """Use LLM to select top-K memories and synthesize guidance"""
        if not candidates:
            return "", []
        
        try:
            # Format candidates for LLM
            candidate_lines = []
            for i, c in enumerate(candidates, 1):
                provenance = c.get("provenance") or {}
                source_type = provenance.get("source_type", "unknown")
                evidence_span = provenance.get("evidence_span", "")
                candidate_lines.append(
                    f"{i}. [{c['type'].upper()}] "
                    f"(Success: {c['success_rate']:.1%}, Harmful: {c['harmful_rate']:.1%}, "
                    f"Confidence: {c['confidence']:.2f}, Source: {source_type})\n"
                    f"   Content: {c['content']}\n"
                    f"   Applies when: {c.get('applies_when') or 'not specified'}\n"
                    f"   Do not apply when: {c.get('do_not_apply_when') or 'not specified'}\n"
                    f"   Evidence: {evidence_span or 'not provided'}"
                )
            
            prompt = f"""You are a memory guidance system. Select and synthesize the most relevant memories for this task.

**Task Query:**
{request.query}

**Available Memories:**
{chr(10).join(candidate_lines)}

**Your Tasks:**
1. Select the top {self.top_k_longterm} most relevant memories (strategic, operational, or failure-pattern cautions)
2. Synthesize them into concise, actionable guidance

**Synthesis Requirements:**
- Keep it within {self.begin_memory_budget_chars} characters
- Include failure-pattern cautions only when their applies_when boundary matches this task
- Combine related points, avoid redundancy
- Use bullet points for readability
- **IMPORTANT: Always frame as suggestions, not commands**: "Consider...", "Based on similar tasks...", "You might want to..."
- Preserve applicability boundaries. If a memory has a "do_not_apply_when" condition that may fit, mention the caveat or skip it.
- Prefer memories with clear provenance, higher confidence, and lower harmful rate.
- Do not turn a task-specific fact into a general rule unless its provenance and applies_when justify it.
**Output Format (JSON):**
{{
  "selected_indices": [1, 3, 5],
  "guidance": "Your synthesized guidance here..."
}}

**Your Response:**"""

            response = self._call_llm(prompt)
            
            if not response or not response.strip():
                return ""
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if result:
                selected_indices = result.get("selected_indices", [])
                guidance = result.get("guidance", "")
                
                # Update usage_count for selected memories
                selected_memory_ids = self._update_memory_usage(candidates, selected_indices)
                
                return self._compact_text(guidance.strip(), self.begin_memory_budget_chars), selected_memory_ids
            else:
                # Fallback: treat response as plain text guidance
                self.logger.warning("Failed to parse JSON from LLM, treating as plain text")
                return self._compact_text(response.strip(), self.begin_memory_budget_chars), []
            
        except Exception as e:
            self.logger.error(f"Long-term synthesis error: {str(e)}", exc_info=True)
            return "", []

    def _update_memory_usage(self, candidates: List[Dict[str, Any]], selected_indices: List[int]) -> List[str]:
        """
        Update usage_count for selected memories and track them for success rate updates
        
        Args:
            candidates: List of memory candidates with id, type, content
            selected_indices: List of selected indices (1-based)
        """
        selected_memory_ids = []
        try:
            for idx in selected_indices:
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
                # Convert to 0-based index
                if idx < 1 or idx > len(candidates):
                    continue
                
                candidate = candidates[idx - 1]
                memory_id = candidate["id"]
                memory_type = candidate["type"]
                selected_memory_ids.append(memory_id)
                
                # Track this memory for success rate updates later
                self.task_context["used_memory_ids"].append(memory_id)
                
                # Update usage_count in the database
                memory_list, mem_idx = self._resolve_memory_ref(memory_id)
                if memory_list is not None and 0 <= mem_idx < len(memory_list):
                    memory_list[mem_idx]["usage_count"] = memory_list[mem_idx].get("usage_count", 0) + 1
                    self.logger.debug(f"Updated usage_count for {memory_type} memory {mem_idx}")
            
            # Save snapshot: preserve used_memory_ids for this task (keyed by query)
            # Always save snapshot (even if empty) to track that this task has been processed
            current_query = self.task_context.get("query", "")
            if current_query:
                query_key = self._compute_signature(current_query)  # Reuse existing hash function
                self._task_snapshots[query_key] = list(self.task_context["used_memory_ids"])
                self.logger.info(
                    f"📸 Saved task snapshot: hash={query_key[:16]}..., "
                    f"{len(self.task_context['used_memory_ids'])} memory IDs, "
                    f"query_preview={current_query[:80]}..."
                )
        
        except Exception as e:
            self.logger.warning(f"Error updating memory usage: {str(e)}")
        
        return selected_memory_ids

    # =========================================================================
    # Short-term Memory Auto-Extraction and Retrieval
    # =========================================================================

    def _calculate_context_delta(self, current_context: str) -> str:
        """
        Calculate the delta between current context and last context
        
        Returns the NEW information that was added in this step (no truncation)
        """
        last_context = self.task_context.get("last_context", "")
        
        if not last_context:
            # First step, return full context
            return current_context.strip()
        
        # Simple delta: remove the prefix that matches last_context
        # This assumes context grows incrementally (common pattern in agent execution)
        current = current_context.strip()
        last = last_context.strip()
        
        if current.startswith(last):
            # Current context includes last context as prefix, extract the delta
            delta = current[len(last):].strip()
            return delta if delta else current
        
        # If not a simple append, return the full current context
        # (fallback for cases where context is restructured)
        return current_context.strip()

    def _auto_extract_shortterm(self, request: MemoryRequest) -> None:
        """
        Automatically extract key information and generate step summary
        
        Combined process:
        1. Generate step summary for execution history
        2. Extract key information for short-term memory
        
        Uses LLM to identify:
        - Important numerical results or data points
        - Task constraints and requirements
        - Critical discoveries or findings
        """
        if not self.model:
            return
        
        try:
            # Calculate context delta (what's new in this step)
            context_delta = self._calculate_context_delta(request.context)
            
            # Update last_context for next delta calculation
            self.task_context["last_context"] = request.context
            
            # Skip if delta is too short (no meaningful new information)
            if len(context_delta.strip()) < 50:
                return
            
            # Build current memory context
            current_memory_str = ""
            if self.shortterm_memory:
                current_memory_str = "\n".join([f"- {item}" for item in self.shortterm_memory])
            else:
                current_memory_str = "(No memory items yet)"
            
            # Build previous steps summary
            prev_steps_summary = self._build_prev_steps_summary()
            #- **If the task is time-sensitive, for time-dependent facts or data (e.g., population in a specific year, event dates), include the relevant time information in the extracted key point**
            # Combined LLM call: Generate step summary + Extract key information
            prompt = f"""You are analyzing the current step of task execution. Perform TWO tasks:

1. Generate a brief summary of what happened in this step
2. Extract key information that should be remembered

**Original Task:**
{request.query}

**Previous Steps Summary:**
{prev_steps_summary}

**Current Working Memory (already remembered):**
{current_memory_str}

**New Information from Current Step:**
{context_delta}

**Output Format (JSON):**
{{
  "step_summary": "2-3 sentence summary of what agent did in this step",
  "key_extracts": ["item1", "item2", "item3"]
}}

**Task 1 - Step Summary Requirements:**
Capture in 2-3 sentences:
- What action was taken (searched, calculated, analyzed, etc.)
- What was discovered or determined
- Any important state changes

**Task 2 - Key Extraction Requirements (maximum 3 items):**
Extract ONLY task-related information that is truly critical:

**EXTRACT (Task-Related Information):**
- Task-specific constraints or requirements discovered during execution (e.g., "output must be in kilometers", "no commas allowed", "date format: YYYY-MM-DD")
- Key data values or numerical results from the task (e.g., "Beijing coordinates: 39.9°N, 116.4°E", "total population: 21.5 million")
- Important discoveries or findings related to the task content (e.g., "source A contradicts source B", "calculation requires unit conversion")
- Task-specific conditions or constraints (e.g., "must use data from 2023", "exclude weekends")

**DO NOT EXTRACT (System/Format Instructions):**
- Memory system guidance
- System instructions or format requirements (e.g., "must include think section", "use tools array", "call final_answer when done")
- Tool usage specifications or API requirements
- General execution guidelines or workflow rules
- Agent behavior instructions or protocol requirements
- Any meta-instructions about how to structure responses or use tools

**Extraction Rules:**
- Each item: ONE concise sentence (15-20 words max), **prefer direct quotes or close paraphrasing from the context to preserve original meaning**
- **ONLY extract information directly related to the TASK CONTENT, not execution mechanics**
- **DO NOT extract information already in working memory**
- Prioritize: Task Constraints > Key Data Values > Important Discoveries
- If nothing new/important related to the task, set "key_extracts" to empty array []
- **Preserve original meaning: quote or closely paraphrase the source text, don't rephrase unnecessarily**

**Critical Filter:**
Before extracting any item, ask: "Is this about the TASK CONTENT (data, constraints, results) or about SYSTEM INSTRUCTIONS (format, tools, workflow)?"
- If about task content → EXTRACT
- If about system instructions → SKIP

**Your Response (JSON only):**"""

            response = self._call_llm(prompt)
            
            if not response or not response.strip():
                return
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if not result:
                self.logger.warning(f"Failed to parse JSON response in auto-extraction: {response[:200]}")
                return
            
            # Extract and store step summary
            step_summary = result.get("step_summary", "").strip()
            if step_summary:
                current_step = self.task_context["current_step"]
                self.task_context["agent_steps"].append({
                    "step": current_step,
                    "summary": step_summary,
                    "phase": request.status.value,
                })
                self.logger.debug(f"Step {current_step} summary: {step_summary[:60]}...")
            
            # Extract and add key information items
            key_extracts = result.get("key_extracts", [])
            if isinstance(key_extracts, list):
                for item in key_extracts:
                    item = str(item).strip()
                    if item and len(item) > 10:
                        self.add_shortterm_item(item)
                        self.logger.debug(f"Auto-extracted short-term item: {item[:60]}...")
            
        except Exception as e:
            self.logger.warning(f"Auto-extraction error: {str(e)}")
            # Non-critical error, continue execution

    def _build_prev_steps_summary(self) -> str:
        """Build a compact summary of previous steps"""
        if not self.task_context["agent_steps"]:
            return "No previous steps (this is the first step)."
        
        summaries = []
        for step_info in self.task_context["agent_steps"]:
            summaries.append(f"Step {step_info['step']}: {step_info['summary']}")
        
        return "\n".join(summaries)

    def _retrieve_shortterm_memory(self, request: MemoryRequest) -> Optional[MemoryItem]:
        """
        Retrieve short-term memory for IN phase
        
        Returns accumulated key facts/constraints from task execution
        """
        if not self.shortterm_memory:
            self.logger.info("No short-term memory to provide")
            return None
        
        try:
            # Format short-term memory into readable text
            content_lines = ["**Key Information & Constraints:**"]
            for idx, item in enumerate(self.shortterm_memory, 1):
                content_lines.append(f"{idx}. {item}")
            
            content = self._compact_text("\n".join(content_lines), self.in_memory_budget_chars)
            
            self._memory_id_counter += 1
            memory_item = MemoryItem(
                id=f"lightweight_shortterm_{self._memory_id_counter}",
                content=content,
                metadata={
                    "source": "lightweight_shortterm",
                    "phase": "in",
                    "item_count": len(self.shortterm_memory),
                    "budget_chars": self.in_memory_budget_chars,
                    "provenance": {
                        "source_url": None,
                        "source_type": "working_memory",
                        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "evidence_span": "Current task trajectory short-term extracts",
                        "confidence": 0.6,
                        "expires_at": None,
                    },
                },
                type=MemoryItemType.TEXT,
            )
            
            return memory_item
            
        except Exception as e:
            self.logger.error(f"Short-term memory retrieval error: {str(e)}", exc_info=True)
            return None

    def add_shortterm_item(self, item: str) -> None:
        """
        Add an item to short-term memory with LLM-based importance ranking
        
        When capacity is reached, LLM decides which items to keep based on importance
        """
        if not item or not item.strip():
            return
        
        item = item.strip()
        
        # Check for duplicates (simple string matching)
        if item in self.shortterm_memory:
            return
        
        # Add new item
        self.shortterm_memory.append(item)
        
        # If within limit, just add
        if len(self.shortterm_memory) <= self.max_shortterm_items:
            self.logger.debug(f"Added short-term item: {item[:60]}...")
            return
        
        # Capacity exceeded: Use LLM to prune least important items
        self.logger.info(f"Short-term memory capacity exceeded ({len(self.shortterm_memory)}/{self.max_shortterm_items}), using LLM to prune...")
        pruned = self._prune_shortterm_memory()
        
        if pruned:
            self.logger.debug(f"Pruned {len(pruned)} items from short-term memory")
            for removed_item in pruned:
                self.logger.debug(f"  - Removed: {removed_item[:60]}...")
        
        self.logger.debug(f"Short-term memory now contains {len(self.shortterm_memory)} items")

    def _prune_shortterm_memory(self) -> List[str]:
        """
        Use LLM to prune short-term memory by removing least important items
        
        Returns: List of removed items
        """
        if not self.model:
            # Fallback: remove oldest items if no model available
            removed = []
            while len(self.shortterm_memory) > self.max_shortterm_items:
                removed.append(self.shortterm_memory.pop(0))
            return removed
        
        try:
            # Format all items for LLM evaluation
            item_lines = []
            for idx, item in enumerate(self.shortterm_memory, 1):
                item_lines.append(f"{idx}. {item}")
            
            task_query = self.task_context.get("query", "Unknown task")
            
            # Build execution history
            prev_steps_summary = self._build_prev_steps_summary()
            
            prompt = f"""You are managing a working memory for an AI agent solving a task. The memory is full and you must decide which items to keep.

**Original Task:**
{task_query}

**Execution History (what agent has done):**
{prev_steps_summary}

**Current Memory Items (newest at bottom):**
{chr(10).join(item_lines)}

**Task:**
Select exactly {self.max_shortterm_items} MOST IMPORTANT items to keep.

**Use the Execution History to judge:**
- What stage is the task at? (early exploration / data gathering / calculation / finalization)
- What information is still needed based on what's been done?
- Which memory items are critical for the NEXT steps?

**Priority Guidelines:**

**ALWAYS KEEP (regardless of task stage):**
- Task constraints: units, format, output requirements
- Key data values that will be used in final answer
- Critical discoveries that changed the approach

**CONSIDER TASK STAGE:**
- Early stage (steps 1-10): Keep constraints and initial data
- Mid stage (steps 11-25): Keep data being actively used, formulas
- Late stage (steps 26+): Keep constraints, final results, discard superseded intermediate results

**CAN DISCARD:**
- Information superseded by newer, more accurate data
- Intermediate steps no longer relevant
- General actions that don't reveal constraints
- Redundant information

**CRITICAL REMINDER:** 
Constraints discovered early (like "no commas", "kilometers only") remain CRITICAL even in late stages because they determine final answer format. Do NOT discard them!

**Output Format (JSON list of indices to KEEP):**
[1, 3, 5, 7, ...]

Only provide the JSON array, nothing else.

**Your Selection:**"""

            response = self._call_llm(prompt)
            
            if not response or not response.strip():
                # Fallback: keep most recent items
                removed = []
                while len(self.shortterm_memory) > self.max_shortterm_items:
                    removed.append(self.shortterm_memory.pop(0))
                return removed
            
            # Parse LLM response
            keep_indices = self._parse_json_response(response)
            if not keep_indices:
                # Try to extract numbers from response as fallback
                numbers = re.findall(r'\d+', response)
                keep_indices = [int(n) for n in numbers]
            
            if not isinstance(keep_indices, list) or len(keep_indices) == 0:
                # Fallback
                removed = []
                while len(self.shortterm_memory) > self.max_shortterm_items:
                    removed.append(self.shortterm_memory.pop(0))
                return removed
            
            # Ensure indices are within range and limit to max_shortterm_items
            keep_indices = [i for i in keep_indices if 1 <= i <= len(self.shortterm_memory)]
            keep_indices = keep_indices[:self.max_shortterm_items]
            
            # Convert to 0-based indices
            keep_set = set(i - 1 for i in keep_indices)
            
            # Separate items to keep and remove
            new_memory = []
            removed = []
            for idx, item in enumerate(self.shortterm_memory):
                if idx in keep_set:
                    new_memory.append(item)
                else:
                    removed.append(item)
            
            # Update memory
            self.shortterm_memory = new_memory
            
            return removed
            
        except Exception as e:
            self.logger.warning(f"Short-term memory pruning error: {str(e)}, using fallback FIFO")
            # Fallback: remove oldest items
            removed = []
            while len(self.shortterm_memory) > self.max_shortterm_items:
                removed.append(self.shortterm_memory.pop(0))
            return removed

    def _resolve_memory_ref(self, memory_id: str) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Resolve a stored memory id such as strategic_3 or failure_pattern_2."""
        if not memory_id or not isinstance(memory_id, str):
            return None, -1
        for prefix, db_key in [
            ("failure_pattern_", "failure_patterns"),
            ("strategic_", "strategic"),
            ("operational_", "operational"),
        ]:
            if memory_id.startswith(prefix):
                try:
                    return self.longterm_db.get(db_key, []), int(memory_id[len(prefix):])
                except ValueError:
                    return None, -1
        return None, -1

    def _memory_confidence(self, memory: Dict[str, Any]) -> float:
        provenance = memory.get("provenance") or {}
        value = memory.get("confidence", provenance.get("confidence", 0.45))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.45

    def _update_memory_feedback_counts(self, memory_feedback: Optional[Dict[str, Any]]) -> None:
        """Apply post-task labels back to the long-term memories that produced guidance."""
        if not memory_feedback:
            return
        events = memory_feedback.get("events") if isinstance(memory_feedback, dict) else None
        if not isinstance(events, list):
            return

        for event in events:
            if not isinstance(event, dict):
                continue
            label = event.get("label")
            metadata = event.get("metadata") or {}
            selected_ids = metadata.get("selected_memory_ids") or []
            if isinstance(selected_ids, str):
                selected_ids = [selected_ids]
            if not selected_ids and event.get("memory_id", "").startswith(("strategic_", "operational_", "failure_pattern_")):
                selected_ids = [event.get("memory_id")]
            
            for memory_id in selected_ids:
                memory_list, mem_idx = self._resolve_memory_ref(memory_id)
                if memory_list is None or not (0 <= mem_idx < len(memory_list)):
                    continue
                memory = memory_list[mem_idx]
                if label in {"helpful", "neutral", "harmful", "stale", "unsupported"}:
                    count_key = f"{label}_count"
                    memory[count_key] = memory.get(count_key, 0) + 1
                memory["last_feedback_label"] = label
                memory["last_feedback_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _update_memory_success_count(self, query: Optional[str] = None) -> None:
        """
        Update success_count for all memories used in this successful task
        
        Args:
            query: Task query to retrieve snapshot (if current context is empty)
        """
        try:
            used_memory_ids = self.task_context.get("used_memory_ids", [])
            
            # Fallback: restore from snapshot if current list is empty
            if not used_memory_ids and query:
                query_key = self._compute_signature(query)
                self.logger.info(
                    f"🔍 Looking for snapshot: hash={query_key[:16]}..., "
                    f"query_preview={query[:80]}..., "
                    f"available_snapshots={list(k[:16] + '...' for k in self._task_snapshots.keys())}"
                )
                used_memory_ids = self._task_snapshots.get(query_key, [])
                if used_memory_ids:
                    self.logger.info(
                        f"✅ Restored {len(used_memory_ids)} memory IDs from task snapshot "
                        f"(query hash={query_key[:16]}...)"
                    )
                else:
                    self.logger.warning(
                        f"❌ Snapshot NOT found for hash={query_key[:16]}..."
                    )
            
            if not used_memory_ids:
                self.logger.warning("No memories were used in this task (neither in context nor snapshot)")
                return
            
            updated_count = 0
            for memory_id in used_memory_ids:
                try:
                    memory_list, mem_idx = self._resolve_memory_ref(memory_id)
                    if memory_list is None or not (0 <= mem_idx < len(memory_list)):
                        continue
                    memory_list[mem_idx]["success_count"] = memory_list[mem_idx].get("success_count", 0) + 1
                    updated_count += 1
                    self.logger.debug(f"Updated success_count for memory {memory_id}")
                
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing memory_id {memory_id}: {str(e)}")
                    continue
            
            self.logger.info(f"Updated success_count for {updated_count} memories")
        
        except Exception as e:
            self.logger.warning(f"Error updating memory success count: {str(e)}")

    # =========================================================================
    # Memory Extraction and Storage
    # =========================================================================

    def _normalize_memory_record(
        self,
        raw_memory: Any,
        memory_type: str,
        default_confidence: float = 0.45,
    ) -> Dict[str, Any]:
        """Normalize legacy strings and new structured memories into one schema."""
        if isinstance(raw_memory, dict):
            content = (
                raw_memory.get("content")
                or raw_memory.get("insight")
                or raw_memory.get("pattern")
                or raw_memory.get("text")
                or ""
            )
            tags = raw_memory.get("tags") or self._extract_tags(str(content))
            provenance = raw_memory.get("provenance") or {}
            confidence = raw_memory.get("confidence", provenance.get("confidence", default_confidence))
            applies_when = raw_memory.get("applies_when")
            do_not_apply_when = raw_memory.get("do_not_apply_when")
        else:
            content = str(raw_memory)
            tags = self._extract_tags(content)
            provenance = {}
            confidence = default_confidence
            applies_when = None
            do_not_apply_when = None

        content = str(content).strip()
        signature = self._compute_signature(content.lower())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if not isinstance(provenance, dict):
            provenance = {}
        if isinstance(raw_memory, dict):
            for field in ["source_url", "source_type", "retrieved_at", "evidence_span", "expires_at"]:
                if raw_memory.get(field) and not provenance.get(field):
                    provenance[field] = raw_memory.get(field)
        provenance = {
            "source_url": provenance.get("source_url"),
            "source_type": provenance.get("source_type") or "trajectory",
            "retrieved_at": provenance.get("retrieved_at") or now,
            "evidence_span": provenance.get("evidence_span", ""),
            "confidence": provenance.get("confidence", confidence),
            "expires_at": provenance.get("expires_at"),
        }

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = default_confidence

        if memory_type == "failure_pattern":
            default_applies = "Use when a new task shows similar uncertainty, tool failure, evidence gaps, or boundary confusion."
            default_do_not_apply = "Do not treat this as a factual conclusion; use it only as a caution/checklist."
        else:
            default_applies = "Use when the task constraints and evidence situation match this guidance."
            default_do_not_apply = "Do not reuse task-specific facts without fresh evidence or matching applicability boundaries."

        record = {
            "content": content,
            "tags": tags,
            "usage_count": 0,
            "success_count": 0,
            "helpful_count": 0,
            "neutral_count": 0,
            "harmful_count": 0,
            "stale_count": 0,
            "unsupported_count": 0,
            "signature": signature,
            "memory_kind": memory_type,
            "applies_when": applies_when or default_applies,
            "do_not_apply_when": do_not_apply_when or default_do_not_apply,
            "provenance": provenance,
            "confidence": confidence,
            "created_at": now,
            "updated_at": now,
        }

        if isinstance(raw_memory, dict):
            for key in [
                "usage_count", "success_count", "helpful_count", "neutral_count",
                "harmful_count", "stale_count", "unsupported_count", "created_at",
                "updated_at", "signature",
            ]:
                if key in raw_memory:
                    record[key] = raw_memory[key]

        return record

    def _extract_memories(self, trajectory_data: TrajectoryData) -> Optional[Dict[str, Any]]:
        """Use LLM to extract strategic and operational memories from trajectory"""
        if not self.model:
            return None
        
        try:
            trajectory_str = json.dumps(trajectory_data.trajectory or [], ensure_ascii=False)
            
            prompt = f"""Extract reusable learnings from this successful execution.

**Task:**
{trajectory_data.query}

**Trajectory:**
{trajectory_str}

**Result:**
{str(trajectory_data.result)}

**Extract TWO types (max 2 each):**

**1. STRATEGIC (Task Planning & Method Selection):**
- When/why to choose specific approaches
- How to break down complex problems
- Decision criteria for method selection
Example: "When a task is making no progress, proactively try to get the answer from other third-party sources—sometimes this is even more effective."

**2. OPERATIONAL (Tool Usage & Edge Case Handling):**
- How to use tools/APIs effectively
- How to handle failures or edge cases
- Specific techniques that worked
Example: "When web_search fails, try alternative query phrasings or search terms separately"

**Output (JSON):**
{{
  "strategic": [
    {{
      "content": "Reusable planning insight.",
      "applies_when": "Specific task/context boundary where this applies.",
      "do_not_apply_when": "Boundary where this would mislead.",
      "provenance": {{
        "source_type": "trajectory",
        "source_url": null,
        "evidence_span": "Short quote or close paraphrase from the trajectory.",
        "confidence": 0.55,
        "expires_at": null
      }}
    }}
  ],
  "operational": [
    {{
      "content": "Reusable execution/tool insight.",
      "applies_when": "Specific execution condition where this applies.",
      "do_not_apply_when": "Boundary where this would mislead.",
      "provenance": {{
        "source_type": "trajectory",
        "source_url": null,
        "evidence_span": "Short quote or close paraphrase from the trajectory.",
        "confidence": 0.55,
        "expires_at": null
      }}
    }}
  ]
}}

**Rules:**
- Focus on REUSABLE patterns (not task-specific data like "Beijing = 39.9°N")
- Strategic = planning/decisions, Operational = execution/handling
- Only valuable insights, skip obvious points
- Focus on the agent itself in the trajectory, ignore the guidance content of the memory system
- Preserve applicability boundaries. A short memory that loses its boundary is worse than no memory.
- Do not store factual claims unless they have source/evidence provenance and an explicit expiration/boundary.

**Your Extraction:**"""

            response = self._call_llm(prompt)
            
            if not response or not response.strip():
                self.logger.warning("LLM returned empty response for memory extraction")
                return None
            
            extracted = self._parse_json_response(response)
            
            # Validate structure
            if extracted and isinstance(extracted, dict):
                return extracted
            
        except Exception as e:
            self.logger.warning(f"Memory extraction error: {str(e)}")
        
        return None

    def _extract_failure_patterns(self, trajectory_data: TrajectoryData) -> List[Dict[str, Any]]:
        """Extract reusable cautions from failed trajectories without storing factual conclusions."""
        if not self.model:
            return []
        
        try:
            trajectory_str = json.dumps(trajectory_data.trajectory or [], ensure_ascii=False)
            metadata = trajectory_data.metadata or {}
            memory_feedback = metadata.get("memory_feedback") or {}
            
            prompt = f"""Extract reusable failure patterns from this failed execution.

**Task:**
{trajectory_data.query}

**Trajectory:**
{trajectory_str}

**Result:**
{str(trajectory_data.result)}

**Memory Feedback:**
{json.dumps(memory_feedback, ensure_ascii=False)}

**Extract up to 3 failure patterns.**

Store only process-level cautions such as:
- wrong assumption
- tool/search failure or missing fallback
- insufficient evidence
- incomplete list/enumeration
- stale/unsupported memory use
- date/version boundary misunderstanding
- lost applicability boundary

Do NOT store factual conclusions from the failed trajectory.

**Output (JSON array):**
[
  {{
    "content": "Failure pattern or caution.",
    "failure_type": "wrong_assumption | tool_failure | evidence_gap | incomplete_list | date_boundary | stale_memory | unsupported_memory | boundary_loss | other",
    "applies_when": "Specific condition where this caution applies.",
    "do_not_apply_when": "Condition where this caution would be irrelevant.",
    "provenance": {{
      "source_type": "failed_trajectory",
      "source_url": null,
      "evidence_span": "Short quote or close paraphrase from the trajectory.",
      "confidence": 0.5,
      "expires_at": null
    }}
  }}
]

**Your Extraction (JSON only):**"""

            response = self._call_llm(prompt)
            if not response or not response.strip():
                return []
            
            extracted = self._parse_json_response(response)
            if isinstance(extracted, list):
                return extracted
            if isinstance(extracted, dict):
                patterns = extracted.get("failure_patterns") or extracted.get("patterns") or []
                return patterns if isinstance(patterns, list) else []
        
        except Exception as e:
            self.logger.warning(f"Failure pattern extraction error: {str(e)}")
        
        return []

    def _add_failure_patterns(self, new_patterns: List[Any]) -> int:
        """Add failure-pattern memories to long-term database."""
        if not new_patterns:
            return 0
        
        self.longterm_db.setdefault("failure_patterns", [])
        added_count = 0
        for raw_pattern in new_patterns:
            record = self._normalize_memory_record(raw_pattern, memory_type="failure_pattern", default_confidence=0.5)
            content = record["content"]
            if len(content) < 20:
                continue
            if self._find_memory_by_signature(self.longterm_db["failure_patterns"], record["signature"]):
                continue
            if isinstance(raw_pattern, dict) and raw_pattern.get("failure_type"):
                record["failure_type"] = raw_pattern.get("failure_type")
                record["tags"] = list(set(record.get("tags", []) + [str(raw_pattern.get("failure_type"))]))
            self.longterm_db["failure_patterns"].append(record)
            added_count += 1
        
        current_count = len(self.longterm_db["failure_patterns"])
        threshold = self.max_failure_patterns + self.memory_buffer_size
        if current_count > threshold:
            self.logger.info(
                f"Failure-pattern memory buffer full ({current_count} > {threshold}), "
                f"pruning to {self.max_failure_patterns}"
            )
            self._intelligent_prune_memories(
                self.longterm_db["failure_patterns"],
                self.max_failure_patterns,
                memory_type="failure_pattern",
            )
        
        return added_count

    def _add_strategic_memories(self, new_memories: List[Any]) -> int:
        """
        Add strategic memories to long-term database
        
        Uses buffer mechanism: allows expansion to (max + buffer) before pruning.
        This gives new memories multiple task cycles to prove their value.
        
        Returns: number of memories added
        """
        if not new_memories:
            return 0
        
        added_count = 0
        
        for raw_memory in new_memories:
            record = self._normalize_memory_record(raw_memory, memory_type="strategic", default_confidence=0.55)
            content = record["content"]
            if len(content) < 20:  # Skip too short
                continue
            
            # Check for duplicates
            signature = record["signature"]
            if self._find_memory_by_signature(self.longterm_db["strategic"], signature):
                continue
            
            self.longterm_db["strategic"].append(record)
            added_count += 1
        
        # Trigger pruning only when exceeding (max + buffer)
        current_count = len(self.longterm_db["strategic"])
        threshold = self.max_strategic_memories + self.memory_buffer_size
        
        if current_count > threshold:
            self.logger.info(
                f"Strategic memory buffer full ({current_count} > {threshold}), "
                f"pruning to {self.max_strategic_memories}"
            )
            self._intelligent_prune_memories(
                self.longterm_db["strategic"], 
                self.max_strategic_memories,
                memory_type="strategic"
            )
        
        return added_count

    def _add_operational_memories(self, new_memories: List[Any]) -> int:
        """
        Add operational memories to long-term database
        
        Uses buffer mechanism: allows expansion to (max + buffer) before pruning.
        This gives new memories multiple task cycles to prove their value.
        
        Returns: number of memories added
        """
        if not new_memories:
            return 0
        
        added_count = 0
        
        for raw_memory in new_memories:
            record = self._normalize_memory_record(raw_memory, memory_type="operational", default_confidence=0.55)
            content = record["content"]
            if len(content) < 20:  # Skip too short
                continue
            
            # Check for duplicates
            signature = record["signature"]
            if self._find_memory_by_signature(self.longterm_db["operational"], signature):
                continue
            
            self.longterm_db["operational"].append(record)
            added_count += 1
        
        # Trigger pruning only when exceeding (max + buffer)
        current_count = len(self.longterm_db["operational"])
        threshold = self.max_operational_memories + self.memory_buffer_size
        
        if current_count > threshold:
            self.logger.info(
                f"Operational memory buffer full ({current_count} > {threshold}), "
                f"pruning to {self.max_operational_memories}"
            )
            self._intelligent_prune_memories(
                self.longterm_db["operational"], 
                self.max_operational_memories,
                memory_type="operational"
            )
        
        return added_count

    def _intelligent_prune_memories(
        self, 
        memory_list: List[Dict[str, Any]], 
        max_size: int,
        memory_type: str
    ) -> None:
        """
        Intelligently prune memory list using LLM to evaluate quality and redundancy
        
        Batch pruning triggered when buffer is full. Evaluates all memories equally
        based on performance metrics, quality, and redundancy.
        
        Criteria:
        1. Performance metrics (success_rate, usage_count)
        2. Content quality and specificity
        3. Redundancy with other memories
        """
        if len(memory_list) <= max_size:
            return
        
        # If no model available, fall back to simple scoring
        if not self.model:
            self._prune_memories_fallback(memory_list, max_size)
            return
        
        try:
            num_to_remove = len(memory_list) - max_size
            
            self.logger.info(
                f"Pruning {memory_type} memories: {len(memory_list)} -> {max_size} "
                f"(removing {num_to_remove} lowest-value memories)"
            )
            
            # Use LLM to select which memories to REMOVE
            indices_to_remove = self._select_memories_to_remove(
                memory_list, num_to_remove, memory_type
            )
            
            # Remove selected memories
            if indices_to_remove:
                # Sort in descending order to avoid index shifting
                indices_to_remove.sort(reverse=True)
                for idx in indices_to_remove:
                    if 0 <= idx < len(memory_list):
                        removed = memory_list.pop(idx)
                        self.logger.debug(f"Intelligently pruned: {removed['content'][:60]}...")
            else:
                # Fallback if LLM fails
                self.logger.warning("LLM pruning failed, using fallback")
                self._prune_memories_fallback(memory_list, max_size)
            
        except Exception as e:
            self.logger.error(f"Intelligent pruning error: {str(e)}, using fallback")
            self._prune_memories_fallback(memory_list, max_size)
    
    def _select_memories_to_remove(
        self, 
        memory_list: List[Dict[str, Any]], 
        num_to_remove: int,
        memory_type: str
    ) -> List[int]:
        """Use LLM to select which memories should be removed"""
        try:
            # Build memory descriptions for LLM
            memory_lines = []
            for idx, mem in enumerate(memory_list):
                usage = mem.get("usage_count", 0)
                success = mem.get("success_count", 0)
                harmful = mem.get("harmful_count", 0)
                unsupported = mem.get("unsupported_count", 0)
                stale = mem.get("stale_count", 0)
                success_rate = (success / usage * 100) if usage > 0 else 0
                
                status = (
                    f"Used:{usage}, Success:{success}, Rate:{success_rate:.0f}%, "
                    f"Harmful:{harmful}, Unsupported:{unsupported}, Stale:{stale}"
                )
                memory_lines.append(
                    f"{idx + 1}. [{status}]\n   {mem['content']}"
                )
            
            prompt = f"""You are managing a memory system that needs to remove low-value memories during batch cleanup.

**Memory Type:** {memory_type.upper()}

**Current Memories ({len(memory_list)} total):**
{chr(10).join(memory_lines)}

**Task:** Select exactly {num_to_remove} memories to REMOVE.

**Evaluation Criteria (in priority order):**

1. **Performance Metrics:**
   - Low usage count = rarely relevant
   - Low success rate = not helpful when used
   - High harmful/unsupported/stale count = risky or poorly supported
   - Zero usage = never proven useful
   - Compare: Memory with 0 usage < Memory with low success rate < Memory with high success rate
   
2. **Content Quality:**
   - Vague or generic advice (e.g., "be careful", "think about edge cases") - REMOVE
   - Too specific/narrow use case that rarely applies - REMOVE
   - Actionable, clear, reusable insights - KEEP
   
3. **Redundancy:**
   - If multiple memories convey similar advice, keep the one with best performance
   - Remove duplicates or overlapping guidance
   
4. **Fair Evaluation:**
   - Memories with 0 usage haven't proven value yet - strong candidates for removal
   - Between two 0-usage memories, judge by content quality
   - Established high-performers (high usage + high success rate) should be kept

**Removal Strategy:**
- Priority 1: Zero-usage memories (haven't proven value)
- Priority 2: Memories with harmful, unsupported, or stale feedback
- Priority 3: Low success rate (used but unhelpful)
- Priority 4: Redundant or vague content
- Keep: High usage + high success rate memories

**Output Format (JSON array of indices to REMOVE):**
[3, 7, 12, 15, 21]

Provide ONLY the JSON array, nothing else.

**Your Selection:**"""

            response = self._call_llm(prompt)
            
            if not response or not response.strip():
                return []
            
            # Parse LLM response
            indices = self._parse_json_response(response)
            if not indices:
                # Try to extract numbers as fallback
                numbers = re.findall(r'\d+', response)
                indices = [int(n) for n in numbers[:num_to_remove]]
            
            if not isinstance(indices, list):
                return []
            
            # Convert to 0-based and validate
            indices_0based = []
            for idx in indices:
                idx_0 = idx - 1  # Convert to 0-based
                if 0 <= idx_0 < len(memory_list):
                    indices_0based.append(idx_0)
            
            # Ensure we have the right number
            if len(indices_0based) < num_to_remove:
                self.logger.warning(
                    f"LLM selected {len(indices_0based)} memories, need {num_to_remove}"
                )
            
            return indices_0based[:num_to_remove]
            
        except Exception as e:
            self.logger.error(f"Memory selection error: {str(e)}")
            return []
    
    def _prune_memories_fallback(self, memory_list: List[Dict[str, Any]], max_size: int) -> None:
        """
        Fallback pruning: simple score-based removal
        
        Score: success_count * 2 + usage_count - harmful_count * 3 - unsupported_count - stale_count
        
        This gives priority to memories that have been successfully used.
        Memories with zero usage will have score 0 and be removed first.
        """
        if len(memory_list) <= max_size:
            return
        
        # Calculate scores
        for mem in memory_list:
            success_count = mem.get("success_count", 0)
            usage_count = mem.get("usage_count", 0)
            harmful_count = mem.get("harmful_count", 0)
            unsupported_count = mem.get("unsupported_count", 0)
            stale_count = mem.get("stale_count", 0)
            mem["_score"] = (
                success_count * 2
                + usage_count
                - harmful_count * 3
                - unsupported_count
                - stale_count
            )
        
        # Sort by score (descending) and keep top max_size
        memory_list.sort(key=lambda x: x["_score"], reverse=True)
        
        # Remove excess memories (lowest scores)
        removed_count = len(memory_list) - max_size
        for _ in range(removed_count):
            removed = memory_list.pop()
            self.logger.debug(f"Pruned (fallback): {removed['content'][:50]}...")
        
        # Clean up temporary score field
        for mem in memory_list:
            mem.pop("_score", None)

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _parse_json_response(self, text: str) -> Optional[Any]:
        """
        Parse JSON from LLM response, handling code block markers like ```json
        
        Args:
            text: Raw text response from LLM
            
        Returns:
            Parsed JSON object, or None if parsing fails
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from ```json ... ``` code block
        # Handle cases with or without newlines after ```json
        json_block_pattern = r'```json\s*\n?(.*?)```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try to extract from ``` ... ``` code block (generic)
        # Handle cases with or without newlines after ```
        generic_block_pattern = r'```[^\n]*\n?(.*?)```'
        match = re.search(generic_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object/array in the text
        # Look for { ... } or [ ... ] patterns
        json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_arr_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        
        for pattern in [json_obj_pattern, json_arr_pattern]:
            matches = re.findall(pattern, text, re.DOTALL)
            for match_text in matches:
                try:
                    return json.loads(match_text)
                except json.JSONDecodeError:
                    continue
        
        # All attempts failed
        return None

    def _load_longterm_db(self) -> Dict[str, Any]:
        """Load long-term memory database and inject cold-start memories if empty"""
        db = None
        
        if os.path.exists(self.longterm_memory_path):
            try:
                with open(self.longterm_memory_path, "r", encoding="utf-8") as f:
                    db = json.load(f)
            except Exception as e:
                self.logger.error(f"Long-term DB load error: {str(e)}")
        
        if db is None:
            db = {
                "strategic": [],
                "operational": [],
                "failure_patterns": [],
                "meta": {"version": 1}
            }
        
        db.setdefault("strategic", [])
        db.setdefault("operational", [])
        db.setdefault("failure_patterns", [])
        db.setdefault("meta", {"version": 1})
        
        for memory_type, db_key in [
            ("strategic", "strategic"),
            ("operational", "operational"),
            ("failure_pattern", "failure_patterns"),
        ]:
            db[db_key] = [
                self._normalize_memory_record(mem, memory_type=memory_type)
                for mem in db.get(db_key, [])
            ]
        db["meta"]["version"] = max(int(db["meta"].get("version", 1)), 2)
        
        # Inject cold-start memories if database is empty
        if (
            len(db.get("strategic", [])) == 0
            and len(db.get("operational", [])) == 0
            and len(db.get("failure_patterns", [])) == 0
        ):
            self.logger.info("Empty memory database detected, injecting cold-start memories...")
            db = self._inject_coldstart_memories(db)
            self.logger.info(
                f"Injected {len(db['strategic'])} strategic, "
                f"{len(db['operational'])} operational cold-start memories"
            )
        
        return db

    def _inject_coldstart_memories(self, db: Dict[str, Any]) -> Dict[str, Any]:
        """Inject cold-start memories into an empty database"""
        for mem in COLDSTART_STRATEGIC_MEMORIES:
            record = self._normalize_memory_record(mem, memory_type="strategic", default_confidence=0.5)
            record["applies_when"] = mem.get("applies_when", record["applies_when"])
            record["do_not_apply_when"] = mem.get("do_not_apply_when", record["do_not_apply_when"])
            db["strategic"].append(record)
        
        for mem in COLDSTART_OPERATIONAL_MEMORIES:
            record = self._normalize_memory_record(mem, memory_type="operational", default_confidence=0.5)
            record["applies_when"] = mem.get("applies_when", record["applies_when"])
            record["do_not_apply_when"] = mem.get("do_not_apply_when", record["do_not_apply_when"])
            db["operational"].append(record)
        
        return db

    def _save_longterm_db(self) -> None:
        """Save long-term memory database"""
        try:
            os.makedirs(os.path.dirname(self.longterm_memory_path) or ".", exist_ok=True)
            with open(self.longterm_memory_path, "w", encoding="utf-8") as f:
                json.dump(self.longterm_db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Long-term DB save error: {str(e)}")

    def _is_trajectory_success(self, trajectory_data: TrajectoryData) -> bool:
        """Determine if trajectory was successful"""
        metadata = trajectory_data.metadata or {}
        
        if metadata.get("is_correct") is True:
            return True
        if metadata.get("task_success") is True:
            return True
        if metadata.get("outcome") == "success":
            return True
        
        return False

    def _calculate_success_rate(self, memory: Dict[str, Any]) -> float:
        """Calculate success rate of a memory"""
        usage_count = memory.get("usage_count", 0)
        success_count = memory.get("success_count", 0)
        
        if usage_count == 0:
            return 0.0
        
        return success_count / usage_count

    def _find_memory_by_signature(
        self, 
        memory_list: List[Dict[str, Any]], 
        signature: str
    ) -> bool:
        """Check if memory with given signature exists in list"""
        for mem in memory_list:
            if mem.get("signature") == signature:
                return True
        return False

    def _compute_signature(self, text: str) -> str:
        """Compute text signature for deduplication"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text content"""
        tags = []
        text_lower = text.lower()
        
        # Domain tags
        if any(kw in text_lower for kw in ["search", "retrieve", "query", "find"]):
            tags.append("search")
        if any(kw in text_lower for kw in ["calculate", "compute", "count", "sum"]):
            tags.append("computation")
        if any(kw in text_lower for kw in ["validate", "verify", "check"]):
            tags.append("validation")
        if any(kw in text_lower for kw in ["error", "fallback", "handle"]):
            tags.append("error_handling")
        if any(kw in text_lower for kw in ["strategy", "plan", "approach"]):
            tags.append("strategy")
        if any(kw in text_lower for kw in ["web", "crawl", "scrape", "url"]):
            tags.append("web")
        if any(kw in text_lower for kw in ["format", "output", "unit"]):
            tags.append("format")
        
        return list(set(tags))

    def _compact_text(self, text: str, max_length: int = 2000) -> str:
        """
        Compact text to maximum length, keeping the end (most recent content)
        
        Args:
            text: Text to compact
            max_length: Maximum character length
            
        Returns:
            Compacted text with ellipsis at the beginning if truncated
        """
        if not text:
            return ""
        
        text = str(text).strip()
        if len(text) <= max_length:
            return text
        
        # Keep the end of the text (most recent/relevant content)
        # Truncate from the beginning
        return "..." + text[-max_length:]

    def reset_task_context(self, task_id: Optional[str] = None, query: Optional[str] = None) -> None:
        """Reset task context for a new task"""
        if task_id is None:
            self._task_id_counter += 1
            task_id = f"task_{self._task_id_counter}"
        
        self.task_context = {
            "task_id": task_id,
            "query": query,
            "start_time": time.time(),
            "current_step": 0,
            "last_shortterm_provision_step": -999,
            "longterm_provided": False,
            "last_context": "",  # Reset context tracking
            "agent_steps": [],  # Reset step summaries
            "used_memory_ids": [],  # Reset used memory tracking
        }
        self.shortterm_memory = []  # Clear short-term memory
        self.logger.debug(f"Task context reset: {self.task_context['task_id']}")

    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current task"""
        duration = time.time() - self.task_context["start_time"] if self.task_context["start_time"] else 0
        
        return {
            "task_id": self.task_context["task_id"],
            "query": self.task_context["query"],
            "duration_seconds": duration,
            "total_steps": self.task_context["current_step"],
            "shortterm_items": len(self.shortterm_memory),
            "longterm_provided": self.task_context["longterm_provided"],
            "agent_steps": self.task_context["agent_steps"],
            "longterm_counts": {
                "strategic": len((self.longterm_db or {}).get("strategic", [])),
                "operational": len((self.longterm_db or {}).get("operational", [])),
                "failure_patterns": len((self.longterm_db or {}).get("failure_patterns", [])),
            },
        }

    def _call_llm(self, prompt: str) -> str:
        """Call LLM model"""
        if not self.model:
            return ""
        
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            content = getattr(response, "content", str(response))
            return content.strip()
        except Exception as e:
            self.logger.error(f"LLM call error: {str(e)}", exc_info=True)
            return ""

