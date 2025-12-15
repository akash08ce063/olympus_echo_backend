import argparse
import asyncio
import json
import os
import sys

# Standard imports with type ignores for missing libraries in this env
import ngrok
from dotenv import load_dotenv
from fixa import Agent, Evaluation, Scenario, Test, TestRunner
from fixa.evaluators import LocalEvaluator

# Load env variables if .env exists
try:
    load_dotenv(override=True)
except ImportError:
    pass


async def main():
    parser = argparse.ArgumentParser(description="Run Fixa Voice Agent Test")
    parser.add_argument("--config_file", required=True, help="Path to the JSON configuration file")

    args = parser.parse_args()

    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load config file: {str(e)}"}))
        sys.exit(1)

    phone_number_to_call = config.get("phone_number_to_call")
    # Priority: Env Var > Config > Default
    twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER") or config.get("twilio_phone_number", "+15554443333")
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")

    if not phone_number_to_call:
        print(json.dumps({"error": "phone_number_to_call is missing in config"}))
        sys.exit(1)

    # DEBUG LOGS
    print(f"DEBUG: Using From Number (Twilio): {twilio_phone_number}")
    print(f"DEBUG: Using To Number: {phone_number_to_call}")

    # 1. Create Agents
    agents_data = config.get("agents", [])
    agents = []
    for a in agents_data:
        agents.append(Agent(name=a["name"], prompt=a["prompt"], voice_id=a.get("voice_id")))

    # 2. Create Scenarios
    scenarios_data = config.get("scenarios", [])
    scenarios = []
    for s in scenarios_data:
        evals = []
        for e in s.get("evaluations", []):
            evals.append(Evaluation(name=e["name"], prompt=e["prompt"]))

        scenarios.append(Scenario(name=s["name"], prompt=s["prompt"], evaluations=evals))

    # 3. Setup Infrastructure (Ngrok + TestRunner)
    # Start ngrok
    port = 8765
    try:
        # ngrok.forward returns a listener object
        listener = await ngrok.forward(port, authtoken=ngrok_token)
        ngrok_url = listener.url()
    except Exception as e:
        print(json.dumps({"error": f"Failed to start ngrok: {str(e)}"}))
        sys.exit(1)

    # Initialize TestRunner
    test_runner = TestRunner(
        port=port,
        ngrok_url=ngrok_url,
        twilio_phone_number=twilio_phone_number,
        evaluator=LocalEvaluator(),
    )

    # 4. Add Tests
    for scenario in scenarios:
        for agent in agents:
            test = Test(scenario=scenario, agent=agent)
            test_runner.add_test(test)

    # 5. Run Tests
    try:
        test_results = await test_runner.run_tests(
            phone_number=phone_number_to_call,
            type=TestRunner.OUTBOUND,
        )

        # Prepare results for JSON output
        serialized_results = []
        if isinstance(test_results, list):
            for res in test_results:
                # Manually construct dict if model_dump/dict fails or isn't present
                res_dict = {}
                if hasattr(res, "model_dump"):
                    res_dict = res.model_dump()
                elif hasattr(res, "dict"):
                    res_dict = res.dict()
                else:
                    # Fallback for plain objects
                    res_dict = {
                        "agent": getattr(res, "agent", {}).name
                        if hasattr(res, "agent") and hasattr(res.agent, "name")
                        else str(getattr(res, "agent", "Unknown")),
                        "scenario": getattr(res, "scenario", {}).name
                        if hasattr(res, "scenario") and hasattr(res.scenario, "name")
                        else str(getattr(res, "scenario", "Unknown")),
                        "passed": getattr(res, "passed", False),
                        "transcript": getattr(res, "transcript", ""),
                        "recording_url": getattr(res, "recording_url", ""),
                        "error": getattr(res, "error", None),
                        "evaluations": getattr(res, "evaluations", {}),
                    }
                serialized_results.append(res_dict)

            print(json.dumps({"results": serialized_results, "passed": all(r.get("passed", False) for r in serialized_results)}))
        else:
            # Single result handling?
            print(json.dumps({"results": [str(test_results)], "passed": False}))

    except Exception as e:
        print(json.dumps({"error": f"Error running tests: {str(e)}"}))
    finally:
        # Cleanup if needed, though script exit handles most
        pass


if __name__ == "__main__":
    asyncio.run(main())
