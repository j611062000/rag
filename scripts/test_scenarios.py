#!/usr/bin/env python3

"""
🧪 Chat-with-PDF Real-World Scenario Testing

This script tests the system's ability to handle:
1. Ambiguous Questions (requiring clarification)
2. PDF-Only Queries (specific paper content)
3. Autonomous Capability (multi-step reasoning)
4. Out-of-Scope Queries (web search routing)
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
from datetime import datetime

class ChatPDFTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def ask_question(self, question: str, session_id: str = "test") -> Dict[str, Any]:
        """Send a question to the API and return the response"""
        async with self.session.post(
            f"{self.base_url}/ask",
            json={"question": question, "session_id": session_id},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}", "detail": await response.text()}

    async def clear_session(self, session_id: str = "test"):
        """Clear the session to avoid context contamination"""
        async with self.session.post(
            f"{self.base_url}/clear",
            json={"session_id": session_id},
            headers={"Content-Type": "application/json"}
        ) as response:
            return response.status == 200

    def validate_response(self, response: Dict[str, Any], expected_behavior: str) -> Dict[str, Any]:
        """Validate response against expected behavior"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "expected": expected_behavior,
            "response": response,
            "passed": False,
            "reason": ""
        }

        if "error" in response:
            result["reason"] = f"API Error: {response['error']}"
            return result

        answer = response.get("answer", "").lower()
        confidence = response.get("confidence", 0)
        route_used = response.get("route_used", "unknown")

        # Validation logic based on expected behavior
        if "clarification" in expected_behavior.lower():
            # Should ask for clarification
            clarification_indicators = [
                "clarification", "clarify", "more specific", "what do you mean",
                "which", "what specifically", "could you specify", "need more information"
            ]
            if any(indicator in answer for indicator in clarification_indicators):
                result["passed"] = True
                result["reason"] = "Successfully requested clarification"
            else:
                result["reason"] = "Did not request clarification for ambiguous question"

        elif "pdf_search" in expected_behavior.lower():
            # Should use PDF search and find specific information
            if route_used == "pdf" or "sources" in response and response["sources"]:
                if confidence > 0.3:  # Should have decent confidence for PDF matches
                    result["passed"] = True
                    result["reason"] = f"Successfully used PDF search (confidence: {confidence})"
                else:
                    result["reason"] = f"Used PDF search but low confidence: {confidence}"
            else:
                result["reason"] = "Did not use PDF search for document-specific query"

        elif "multi_step" in expected_behavior.lower():
            # Should handle multi-step autonomous reasoning
            if route_used == "both" or "state-of-the-art" in answer or "authors" in answer:
                result["passed"] = True
                result["reason"] = "Successfully handled multi-step reasoning"
            else:
                result["reason"] = "Did not demonstrate multi-step autonomous capability"

        elif "web_search" in expected_behavior.lower():
            # Should route to web search for out-of-scope queries
            if route_used == "web" or confidence > 0.3:
                result["passed"] = True
                result["reason"] = f"Successfully routed to web search"
            else:
                result["reason"] = "Did not route to web search for out-of-scope query"

        return result

    async def run_test_scenario(self, test_case: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Run a single test scenario"""
        print(f"🧪 Testing: {test_case['name']}")
        print(f"   Question: {test_case['question']}")
        print(f"   Expected: {test_case['expected_behavior']}")

        # Clear session before each test
        await self.clear_session(session_id)

        # Wait a moment for session to clear
        await asyncio.sleep(0.5)

        # Ask the question
        response = await self.ask_question(test_case["question"], session_id)

        # Validate the response
        result = self.validate_response(response, test_case["expected_behavior"])
        result.update({
            "test_name": test_case["name"],
            "question": test_case["question"],
            "category": test_case["category"]
        })

        # Print result
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"   Result: {status} - {result['reason']}")
        if not result["passed"]:
            print(f"   Answer: {response.get('answer', 'No answer')[:100]}...")
        print()

        return result

    async def run_all_tests(self):
        """Run all test scenarios"""
        test_cases = [
            # 1. Ambiguous Questions
            {
                "name": "Ambiguous - Vague 'Enough'",
                "category": "ambiguous",
                "question": "How many examples are enough for good accuracy?",
                "expected_behavior": "Should request clarification about dataset and accuracy target"
            },
            {
                "name": "Ambiguous - Vague 'It'",
                "category": "ambiguous",
                "question": "Tell me more about it",
                "expected_behavior": "Should request clarification about what 'it' refers to"
            },

            # 2. PDF-Only Queries
            {
                "name": "PDF Query - Zhang et al. Prompt Template",
                "category": "pdf_search",
                "question": "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?",
                "expected_behavior": "Should find SimpleDDL-MD-Chat as top zero-shot template (65-72% EX)"
            },
            {
                "name": "PDF Query - Davinci-codex Execution",
                "category": "pdf_search",
                "question": "What execution accuracy does davinci-codex reach on Spider with the 'Create Table + Select 3' prompt?",
                "expected_behavior": "Should find 67% execution accuracy for davinci-codex"
            },

            # 3. Autonomous Capability
            {
                "name": "Autonomous - Multi-step State-of-art",
                "category": "multi_step",
                "question": "What's the state-of-the-art text-to-sql approach? And search on the web to tell me more about the authors who contributed to the approach",
                "expected_behavior": "Should autonomously: 1) search PDFs for SOTA approach, 2) find authors, 3) web search for author info"
            },

            # 4. Out-of-Scope Queries
            {
                "name": "Out-of-scope - Current OpenAI",
                "category": "web_search",
                "question": "What did OpenAI release this month?",
                "expected_behavior": "Should recognize out-of-scope and route to web search"
            }
        ]

        print("🚀 Starting Chat-with-PDF Real-World Scenario Tests")
        print("=" * 60)

        # Test health endpoint first
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    print("❌ Service health check failed!")
                    return
                print("✅ Service is healthy, starting tests...\n")
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            return

        # Run all test scenarios
        for i, test_case in enumerate(test_cases, 1):
            session_id = f"test_session_{i}"
            result = await self.run_test_scenario(test_case, session_id)
            self.results.append(result)

            # Small delay between tests
            await asyncio.sleep(1)

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate a comprehensive test report"""
        print("📊 TEST REPORT")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()

        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if result["passed"]:
                categories[cat]["passed"] += 1

        print("📋 Results by Category:")
        for category, stats in categories.items():
            success_rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {category.title()}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")

        print()

        # Failed tests details
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  • {result['test_name']}: {result['reason']}")
            print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if any(not r["passed"] and r["category"] == "ambiguous" for r in self.results):
            print("  • Improve clarification agent logic for ambiguous questions")
        if any(not r["passed"] and r["category"] == "pdf_search" for r in self.results):
            print("  • Enhance PDF search accuracy and document retrieval")
        if any(not r["passed"] and r["category"] == "multi_step" for r in self.results):
            print("  • Implement better multi-step autonomous reasoning")
        if any(not r["passed"] and r["category"] == "web_search" for r in self.results):
            print("  • Improve routing logic for out-of-scope queries")

        # Save detailed results
        with open("test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed results saved to: test_results.json")


async def main():
    """Main test runner"""
    async with ChatPDFTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    print("🧪 Chat-with-PDF Real-World Scenario Tester")
    print("Testing system capabilities for production scenarios...")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")