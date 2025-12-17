'use client';

import { Button } from '@/components/ui/button';
import { useTestStore } from '@/src/store/useTestStore';
import { Play } from 'lucide-react';
import { useState } from 'react';

export function TestRunnerControl() {
    const {
        selectedAgentId,
        testerAgents,
        scenarios,
        setTestResults,
        setIsLoading,
        clearLogs,
        addLog
    } = useTestStore();

    const [isRunning, setIsRunning] = useState(false);

    const handleRunTest = async () => {
        if (!selectedAgentId) {
             // Should be handled by validation ideally
             return;
        }

        setIsRunning(true);
        setIsLoading(true);
        setTestResults(null);
        clearLogs();
        addLog('Starting test suite execution...');

        try {
            const response = await fetch('/api/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dummy: true, // Keeping existing logic
                    agentId: selectedAgentId,
                    testers: testerAgents,
                    scenarios: scenarios
                })
            });

            const data = await response.json();

            if (response.ok) {
                setTestResults(data);
                addLog('Test suite completed successfully.');
            } else {
                console.error('Test run failed:', data.error);
                addLog(`ERROR: Test run failed: ${data.error}`);
            }
        } catch (error) {
            console.error('Network error during test:', error);
            addLog(`ERROR: Network error during test execution.`);
        } finally {
            setIsLoading(false);
            setIsRunning(false);
        }
    };

    const isValid = selectedAgentId && testerAgents.length > 0 && scenarios.length > 0;

    return (
        <div className="flex flex-col gap-4">
             <Button
                size="lg"
                className="w-full text-lg h-14 font-semibold shadow-lg hover:shadow-xl transition-all"
                onClick={handleRunTest}
                disabled={isRunning || !isValid}
            >
                {isRunning ? (
                    <>
                        <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-white mr-2"></div>
                        Running Tests...
                    </>
                ) : (
                    <>
                        <Play className="w-5 h-5 mr-2 fill-current" />
                        Run Test Suite
                    </>
                )}
            </Button>
            {!isValid && (
                 <p className="text-xs text-center text-red-500 animate-pulse">
                     Select a target agent, at least one tester, and one scenario to run.
                 </p>
            )}
        </div>
    );
}
