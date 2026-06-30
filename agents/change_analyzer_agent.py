"""
agents/change_analyzer_agent.py
--------------------------------
ChangeAnalyzerAgent: Parses git diff and extracts structured features.

Uses AutoGen AssistantAgent for change_type classification (LLM call).
All other features extracted programmatically — no LLM tokens wasted.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import re
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Module-to-file mapping (simulates a real codebase)
MODULE_FILE_MAP = {
    "payment_service":    ["payment", "billing", "transaction", "checkout"],
    "auth_service":       ["auth", "login", "token", "session", "oauth"],
    "user_management":    ["user", "profile", "account", "registration"],
    "order_processing":   ["order", "cart", "fulfillment", "shipment"],
    "inventory_service":  ["inventory", "stock", "warehouse", "product"],
    "notification_service": ["notification", "email", "sms", "push"],
    "reporting_engine":   ["report", "analytics", "dashboard", "metrics"],
    "api_gateway":        ["gateway", "router", "middleware", "proxy"],
    "data_pipeline":      ["pipeline", "etl", "transform", "ingest"],
    "search_service":     ["search", "index", "query", "elastic"],
}

CHANGE_TYPES = ["bugfix", "feature", "refactor", "hotfix", "chore"]


class ChangeAnalyzerAgent:
    """
    AutoGen-style agent that analyzes git diffs.

    Runs as a direct agent (Option B) — no GroupChat overhead.
    LLM is called only for change_type classification.
    Everything else is deterministic parsing.
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, use LLM for change_type classification.
                     If False, use keyword heuristic (no API cost).
        """
        self.name = "ChangeAnalyzerAgent"
        self.use_llm = use_llm

        if use_llm:
            self._init_autogen()

    def _init_autogen(self):
        """Initialize AutoGen AssistantAgent for LLM calls."""
        try:
            import autogen
            config = [{
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
            }]
            self.agent = autogen.AssistantAgent(
                name=self.name,
                llm_config={"config_list": config, "max_tokens": 50},
                system_message=(
                    "You classify git commit changes. "
                    "Respond with exactly one word from: bugfix, feature, refactor, hotfix, chore. "
                    "Nothing else."
                ),
            )
        except Exception:
            self.use_llm = False

    def analyze(self, diff: str) -> Dict:
        """
        Parse git diff and return structured feature dictionary.

        Args:
            diff: Raw git diff string or simulated diff dict

        Returns:
            {
                files_changed: list[str],
                modules_affected: list[str],
                churn_score: float,       # 0-1
                dependency_depth: int,    # 1-5
                change_type: str,         # bugfix|feature|refactor|hotfix|chore
                lines_added: int,
                lines_removed: int,
            }
        """
        # Handle dict input (simulated diffs in tests)
        if isinstance(diff, dict):
            return self._process_dict_diff(diff)

        files_changed = self._extract_files(diff)
        lines_added, lines_removed = self._count_lines(diff)
        modules_affected = self._map_to_modules(files_changed)
        churn_score = self._compute_churn(lines_added, lines_removed)
        dependency_depth = self._estimate_depth(modules_affected)
        change_type = self._classify_change_type(diff)

        result = {
            "files_changed": files_changed,
            "modules_affected": modules_affected,
            "churn_score": round(churn_score, 3),
            "dependency_depth": dependency_depth,
            "change_type": change_type,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        }

        print(f"[{self.name}] Analyzed diff: "
              f"{len(files_changed)} files, "
              f"modules={modules_affected}, "
              f"churn={churn_score:.2f}, "
              f"type={change_type}")
        return result

    def _extract_files(self, diff: str) -> List[str]:
        """Extract changed file paths from diff header lines."""
        pattern = r"^diff --git a/(.+?) b/.+$"
        files = re.findall(pattern, diff, re.MULTILINE)
        if not files:
            # Fallback: look for +++ lines
            files = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
        return files if files else ["unknown_file.py"]

    def _count_lines(self, diff: str) -> tuple:
        """Count added and removed lines."""
        added = len(re.findall(r"^\+[^+]", diff, re.MULTILINE))
        removed = len(re.findall(r"^-[^-]", diff, re.MULTILINE))
        return added, removed

    def _map_to_modules(self, files: List[str]) -> List[str]:
        """Map file paths to module names."""
        affected = set()
        for filepath in files:
            filepath_lower = filepath.lower()
            for module, keywords in MODULE_FILE_MAP.items():
                if any(kw in filepath_lower for kw in keywords):
                    affected.add(module)
        # Default if no match
        if not affected:
            affected.add("api_gateway")
        return list(affected)

    def _compute_churn(self, added: int, removed: int) -> float:
        """Normalize churn to 0-1 scale."""
        total = added + removed
        if total == 0:
            return 0.1
        # Log scale: 1 line = ~0.05, 100 lines = ~0.5, 500+ lines = ~1.0
        import math
        return min(1.0, math.log1p(total) / math.log1p(500))

    def _estimate_depth(self, modules: List[str]) -> int:
        """
        Estimate dependency depth based on which modules changed.
        Core infrastructure modules have deeper dependency trees.
        """
        depth_map = {
            "api_gateway": 5,
            "auth_service": 4,
            "data_pipeline": 4,
            "payment_service": 3,
            "order_processing": 3,
            "user_management": 2,
            "inventory_service": 2,
            "reporting_engine": 1,
            "notification_service": 1,
            "search_service": 2,
        }
        if not modules:
            return 1
        return max(depth_map.get(m, 1) for m in modules)

    def _classify_change_type(self, diff: str) -> str:
        """
        Classify change type. Uses LLM if available, else keyword heuristic.
        """
        if self.use_llm and hasattr(self, "agent"):
            try:
                import autogen
                proxy = autogen.UserProxyAgent(
                    name="proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False,
                )
                # Send only a short summary to minimize tokens
                summary = diff[:300] if len(diff) > 300 else diff
                proxy.initiate_chat(
                    self.agent,
                    message=f"Classify this change: {summary}",
                    silent=True,
                )
                last = proxy.last_message()
                if last and last.get("content"):
                    word = last["content"].strip().lower().split()[0]
                    if word in CHANGE_TYPES:
                        return word
            except Exception:
                pass

        # Keyword heuristic fallback
        diff_lower = diff.lower()
        if any(w in diff_lower for w in ["fix", "bug", "patch", "hotfix"]):
            return "bugfix"
        if any(w in diff_lower for w in ["refactor", "rename", "restructure"]):
            return "refactor"
        if any(w in diff_lower for w in ["chore", "lint", "format", "upgrade"]):
            return "chore"
        return "feature"

    def _process_dict_diff(self, diff: dict) -> Dict:
        """Handle pre-parsed diff dicts (used in testing)."""
        return {
            "files_changed": diff.get("files_changed", []),
            "modules_affected": diff.get("modules_affected", ["api_gateway"]),
            "churn_score": diff.get("churn_score", 0.5),
            "dependency_depth": diff.get("dependency_depth", 2),
            "change_type": diff.get("change_type", "feature"),
            "lines_added": diff.get("lines_added", 10),
            "lines_removed": diff.get("lines_removed", 5),
        }


def simulate_git_diff(
    modules: List[str] = None,
    change_type: str = "feature",
    churn: str = "medium",
) -> str:
    """
    Generate a realistic simulated git diff for testing.

    Args:
        modules: List of module names to include in diff
        change_type: Type of change to simulate
        churn: 'low' | 'medium' | 'high'
    """
    if modules is None:
        modules = ["payment_service"]

    churn_lines = {"low": 10, "medium": 50, "high": 200}
    n_lines = churn_lines.get(churn, 50)

    diff_parts = []
    for module in modules:
        keywords = MODULE_FILE_MAP.get(module, ["service"])
        filename = f"src/{module}/{keywords[0]}_handler.py"
        added = "\n".join([f"+    line_{i} = process()" for i in range(n_lines // 2)])
        removed = "\n".join([f"-    old_line_{i} = legacy()" for i in range(n_lines // 2)])
        diff_parts.append(
            f"diff --git a/{filename} b/{filename}\n"
            f"--- a/{filename}\n"
            f"+++ b/{filename}\n"
            f"@@ -1,{n_lines//2} +1,{n_lines//2} @@\n"
            f"{removed}\n{added}"
        )

    return "\n".join(diff_parts)


if __name__ == "__main__":
    agent = ChangeAnalyzerAgent(use_llm=False)
    diff = simulate_git_diff(
        modules=["payment_service", "auth_service"],
        change_type="bugfix",
        churn="medium",
    )
    result = agent.analyze(diff)
    print("\nResult:", result)
