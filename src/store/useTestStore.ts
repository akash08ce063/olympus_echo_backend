import { create } from 'zustand';

export interface Agent {
  id: string
  name: string
  phoneNumber: string
}

export interface TesterAgent {
  id: string
  name: string
  prompt: string
  voice_id?: string
}

export interface EvaluationCriteria {
  id: string
  name: string
  prompt: string
}

export interface Scenario {
  id: string
  name: string
  prompt: string
  evaluations: EvaluationCriteria[]
}

export interface TestResult {
  passed: boolean
  results: {
    agent: string
    scenario: string
    transcript: string
    evaluations: Record<string, { passed: boolean; reasoning: string }>
  }[]
}

interface TestStore {
  agents: Agent[]
  selectedAgentId: string | null

  testerAgents: TesterAgent[]
  scenarios: Scenario[]

  testResults: TestResult | null
  isLoading: boolean
  logs: string[]

  setAgents: (agents: Agent[]) => void
  setSelectedAgentId: (id: string) => void

  addTesterAgent: (agent: TesterAgent) => void
  removeTesterAgent: (id: string) => void
  updateTesterAgent: (id: string, agent: Partial<TesterAgent>) => void

  addScenario: (scenario: Scenario) => void
  removeScenario: (id: string) => void
  updateScenario: (id: string, scenario: Partial<Scenario>) => void

  setTestResults: (results: TestResult | null) => void
  setIsLoading: (isLoading: boolean) => void
  addLog: (log: string) => void
  clearLogs: () => void
}

export const useTestStore = create<TestStore>((set) => ({
  agents: [],
  selectedAgentId: null,

  testerAgents: [
    {
      id: "default-tester",
      name: "Jessica",
      prompt: "You are a young woman named Jessica who says 'like' a lot.",
      voice_id: "b7d50908-b17c-442d-ad8d-810c63997ed9"
    }
  ],
  scenarios: [
    {
      id: "default-scenario",
      name: "Order Donut",
      prompt: "Order a dozen donuts with sprinkles and a coffee.",
      evaluations: [
        { id: "eval-1", name: "order_success", prompt: "the order was successful" }
      ]
    }
  ],

  testResults: null,
  isLoading: false,
  logs: [],

  setAgents: (agents) => set({ agents }),
  setSelectedAgentId: (id) => set({ selectedAgentId: id }),

  addTesterAgent: (agent) =>
    set((state) => ({ testerAgents: [...state.testerAgents, agent] })),
  removeTesterAgent: (id) =>
    set((state) => ({ testerAgents: state.testerAgents.filter((a) => a.id !== id) })),
  updateTesterAgent: (id, agent) =>
    set((state) => ({
      testerAgents: state.testerAgents.map((a) =>
        a.id === id ? { ...a, ...agent } : a
      )
    })),

  addScenario: (scenario) =>
    set((state) => ({ scenarios: [...state.scenarios, scenario] })),
  removeScenario: (id) =>
    set((state) => ({ scenarios: state.scenarios.filter((s) => s.id !== id) })),
  updateScenario: (id, scenario) =>
    set((state) => ({
      scenarios: state.scenarios.map((s) =>
        s.id === id ? { ...s, ...scenario } : s
      )
    })),

  setTestResults: (results) => set({ testResults: results }),
  setIsLoading: (isLoading) => set({ isLoading }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  clearLogs: () => set({ logs: [] }),
}))

