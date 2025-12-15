'use client'

import { ThemeToggle } from '@/components/theme-toggle';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { useTestStore } from '@/store/useTestStore';
import { Plus, Terminal, Trash2, X } from 'lucide-react';
import { useEffect } from 'react';

export default function TestPage() {
  const {
    agents,
    selectedAgentId,
    testerAgents,
    scenarios,
    testResults,
    isLoading,
    setAgents,
    setSelectedAgentId,
    addTesterAgent,
    removeTesterAgent,
    updateTesterAgent,
    addScenario,
    removeScenario,
    updateScenario,
    setTestResults,
    setIsLoading
  } = useTestStore()

  // Fetch agents on mount
  useEffect(() => {
    fetch('/api/agents')
      .then(res => res.json())
      .then(data => {
        if (!data.error) {
          setAgents(data)
        }
      })
      .catch(err => console.error('Failed to fetch agents:', err))
  }, [setAgents])

  const handleRunTest = async () => {
    if (!selectedAgentId) return
    const selectedAgent = agents.find(a => a.id === selectedAgentId)
    if (!selectedAgent) return

    setIsLoading(true)
    setTestResults(null)

    try {
      const response = await fetch('/api/run-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agentId: selectedAgentId,
          agentPhoneNumber: selectedAgent.phoneNumber,
          testerAgents,
          scenarios
        })
      })

      const data = await response.json()

      if (response.ok) {
        setTestResults(data)
      } else {
        console.error('Test run failed:', data.error)
        alert(`Test failed: ${data.error}`)
      }
    } catch (error) {
      console.error('Network error during test:', error)
      alert('Network error')
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddTesterAgent = () => {
    addTesterAgent({
      id: crypto.randomUUID(),
      name: `Agent ${testerAgents.length + 1}`,
      prompt: '',
      voice_id: 'b7d50908-b17c-442d-ad8d-810c63997ed9'
    })
  }

  const handleAddScenario = () => {
    addScenario({
      id: crypto.randomUUID(),
      name: `Scenario ${scenarios.length + 1}`,
      prompt: '',
      evaluations: []
    })
  }

  const handleAddEvaluationToScenario = (scenarioId: string) => {
    const scenario = scenarios.find(s => s.id === scenarioId)
    if (!scenario) return

    updateScenario(scenarioId, {
        evaluations: [
            ...scenario.evaluations,
            { id: crypto.randomUUID(), name: '', prompt: '' }
        ]
    })
  }

  const handleUpdateEvaluation = (scenarioId: string, evalId: string, field: 'name' | 'prompt', value: string) => {
      const scenario = scenarios.find(s => s.id === scenarioId)
      if (!scenario) return

      const updatedEvals = scenario.evaluations.map(e =>
          e.id === evalId ? { ...e, [field]: value } : e
      )
      updateScenario(scenarioId, { evaluations: updatedEvals })
  }

    const handleRemoveEvaluation = (scenarioId: string, evalId: string) => {
      const scenario = scenarios.find(s => s.id === scenarioId)
      if (!scenario) return

      const updatedEvals = scenario.evaluations.filter(e => e.id !== evalId)
      updateScenario(scenarioId, { evaluations: updatedEvals })
  }


  return (
    <div className="container mx-auto py-10 px-4 space-y-8 max-w-7xl">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0">
        <div className="flex flex-col space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Fixa Control Center</h1>
          <p className="text-muted-foreground">
            Configure and run multi-agent voice tests.
          </p>
        </div>
        <ThemeToggle />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Configuration */}
        <div className="space-y-6">

          {/* 1. Select Agent */}
          <Card>
            <CardHeader>
              <CardTitle>1. Select Target Agent</CardTitle>
              <CardDescription>The agent you want to test</CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={selectedAgentId || ''} onValueChange={setSelectedAgentId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select an agent..." />
                </SelectTrigger>
                <SelectContent>
                  {agents.map((agent) => (
                    <SelectItem key={agent.id} value={agent.id}>
                      {agent.name} ({agent.phoneNumber})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* 2. Tester Agents */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="space-y-1">
                <CardTitle>2. Tester Agents</CardTitle>
                <CardDescription>Define the personas that will call</CardDescription>
              </div>
              <Button size="sm" variant="outline" onClick={handleAddTesterAgent}>
                <Plus className="w-4 h-4 mr-2" />
                Add Agent
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              {testerAgents.map((agent) => (
                <div key={agent.id} className="p-4 border rounded-md relative space-y-3">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-destructive"
                        onClick={() => removeTesterAgent(agent.id)}
                    >
                        <Trash2 className="w-4 h-4" />
                    </Button>
                    <div className="grid grid-cols-2 gap-2">
                         <div className="space-y-1">
                            <label className="text-xs font-medium">Name</label>
                            <Input
                                value={agent.name}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateTesterAgent(agent.id, { name: e.target.value })}
                                placeholder="Agent Name"
                            />
                         </div>
                         <div className="space-y-1">
                            <label className="text-xs font-medium">Voice ID (Optional)</label>
                             <Input
                                value={agent.voice_id || ''}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateTesterAgent(agent.id, { voice_id: e.target.value })}
                                placeholder="UUID (e.g. b7d5...)"
                            />
                         </div>
                    </div>
                    <div className="space-y-1">
                         <label className="text-xs font-medium">Persona Prompt</label>
                        <Textarea
                            value={agent.prompt}
                            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateTesterAgent(agent.id, { prompt: e.target.value })}
                            placeholder="You are a caller who..."
                            className="min-h-[80px]"
                        />
                    </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* 3. Scenarios */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="space-y-1">
                <CardTitle>3. Scenarios</CardTitle>
                <CardDescription>What situations should be tested?</CardDescription>
              </div>
              <Button size="sm" variant="outline" onClick={handleAddScenario}>
                <Plus className="w-4 h-4 mr-2" />
                Add Scenario
              </Button>
            </CardHeader>
            <CardContent className="space-y-6">
                {scenarios.map((scenario) => (
                    <div key={scenario.id} className="p-4 border rounded-md relative space-y-3">
                         <Button
                            variant="ghost"
                            size="icon"
                            className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-destructive"
                            onClick={() => removeScenario(scenario.id)}
                        >
                            <Trash2 className="w-4 h-4" />
                        </Button>

                        <div className="space-y-1">
                             <label className="text-xs font-medium">Scenario Name</label>
                             <Input
                                value={scenario.name}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateScenario(scenario.id, { name: e.target.value })}
                                placeholder="e.g. Order Pizza"
                             />
                        </div>
                        <div className="space-y-1">
                             <label className="text-xs font-medium">Instruction Prompt</label>
                             <Textarea
                                value={scenario.prompt}
                                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateScenario(scenario.id, { prompt: e.target.value })}
                                placeholder="e.g. Call and order a pepperoni pizza..."
                                className="min-h-[60px]"
                             />
                        </div>

                         <Separator className="my-2"/>

                         <div className="space-y-2">
                            <div className="flex justify-between items-center">
                                <label className="text-xs font-semibold uppercase text-muted-foreground">Evaluation Criteria</label>
                                <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => handleAddEvaluationToScenario(scenario.id)}>
                                    <Plus className="w-3 h-3 mr-1"/> Add Eval
                                </Button>
                            </div>

                            {scenario.evaluations.map((evalItem) => (
                                <div key={evalItem.id} className="flex gap-2 items-start">
                                    <Input
                                        className="h-8 text-xs w-1/3"
                                        placeholder="Name"
                                        value={evalItem.name}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleUpdateEvaluation(scenario.id, evalItem.id, 'name', e.target.value)}
                                    />
                                    <Input
                                        className="h-8 text-xs flex-1"
                                        placeholder="Success condition..."
                                        value={evalItem.prompt}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleUpdateEvaluation(scenario.id, evalItem.id, 'prompt', e.target.value)}
                                    />
                                     <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-8 w-8 text-muted-foreground hover:text-destructive shrink-0"
                                        onClick={() => handleRemoveEvaluation(scenario.id, evalItem.id)}
                                    >
                                        <X className="w-3 h-3" />
                                    </Button>
                                </div>
                            ))}
                            {scenario.evaluations.length === 0 && (
                                <p className="text-xs text-muted-foreground italic">No evaluations defined.</p>
                            )}
                         </div>
                    </div>
                ))}
            </CardContent>
            <CardFooter>
               <Button
                className="w-full"
                size="lg"
                onClick={handleRunTest}
                disabled={isLoading || !selectedAgentId || testerAgents.length === 0 || scenarios.length === 0}
              >
                {isLoading ? 'Running Test Suite...' : 'Run Test Suite'}
              </Button>
            </CardFooter>
          </Card>
        </div>

        {/* Right Column: Results */}
        <div className="space-y-6">
          <Card className="h-full flex flex-col min-h-[600px]">
             <CardHeader>
              <CardTitle>Results & Logs</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-auto">
              {!testResults && !isLoading && (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <Terminal className="w-12 h-12 opacity-20" />
                  <p>Run a test to see results here</p>
                </div>
              )}

              {isLoading && (
                <div className="h-full flex flex-col items-center justify-center space-y-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                  <p className="text-muted-foreground">Test in progress. This may take up to a minute...</p>
                </div>
              )}

              {testResults && (
                <div className="space-y-8 animate-in fade-in duration-500">
                  {/* Overall Status */}
                  <div className="flex items-center justify-between p-4 rounded-lg">
                    <span className="font-semibold">Overall Status</span>
                    <Badge variant={testResults.passed ? "default" : "destructive"} className="text-base px-4 py-1">
                      {testResults.passed ? 'PASSED' : 'FAILED'}
                    </Badge>
                  </div>

                  <Separator />

                  <div className="space-y-6">
                      {testResults.results.map((result, idx) => (
                          <div key={idx} className="border rounded-lg p-4 space-y-4 bg-white dark:bg-black">
                              <div className="flex items-center justify-between">
                                  <div>
                                      <h3 className="font-semibold">{result.agent} <span className="text-muted-foreground text-sm font-normal">testing</span> {result.scenario}</h3>
                                  </div>
                              </div>

                              <div className="space-y-2">
                                  <h4 className="text-sm font-medium text-muted-foreground">Evaluations</h4>
                                  <div className="grid grid-cols-1 gap-2">
                                      {Object.entries(result.evaluations).map(([name, evalRes]) => (
                                          <div key={name} className="flex items-start justify-between p-2 rounded border">
                                              <div>
                                                  <span className="font-medium text-sm">{name}</span>
                                                  <p className="text-xs text-muted-foreground">{evalRes.reasoning}</p>
                                              </div>
                                              <Badge variant={evalRes.passed ? "outline" : "destructive"} className="ml-2">
                                                  {evalRes.passed ? 'Pass' : 'Fail'}
                                              </Badge>
                                          </div>
                                      ))}
                                  </div>
                              </div>

                              <div className="space-y-2">
                                <h4 className="text-sm font-medium text-muted-foreground">Transcript</h4>
                                <div className="p-3 rounded-md bg-slate-950 text-slate-100 text-xs font-mono whitespace-pre-wrap leading-relaxed max-h-[200px] overflow-y-auto">
                                    {result.transcript}
                                </div>
                              </div>
                          </div>
                      ))}
                  </div>

                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
