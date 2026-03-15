"""Kiki SLM evaluation test suites."""

from kiki.evaluation.test_suites.test_intent import IntentTestSuite
from kiki.evaluation.test_suites.test_safety import SafetyTestSuite
from kiki.evaluation.test_suites.test_tool_calling import ToolCallingTestSuite
from kiki.evaluation.test_suites.test_workflow import WorkflowTestSuite

__all__ = ["IntentTestSuite", "SafetyTestSuite", "ToolCallingTestSuite", "WorkflowTestSuite"]
