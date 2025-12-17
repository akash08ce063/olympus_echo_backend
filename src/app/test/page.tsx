'use client';

import { LogStreamViewer } from '@/components/fixa/LogStreamViewer';
import { ScenariosList } from '@/components/fixa/ScenariosList';
import { TargetAgentSelector } from '@/components/fixa/TargetAgentSelector';
import { TesterAgentsList } from '@/components/fixa/TesterAgentsList';
import { TestResultsView } from '@/components/fixa/TestResultsView';
import { TestRunnerControl } from '@/components/fixa/TestRunnerControl';
import { ThemeToggle } from '@/components/theme-toggle';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { motion } from 'framer-motion';

export default function TestPage() {
    return (
        <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
            <div className="container mx-auto py-8 px-4 sm:px-6 lg:px-8 space-y-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0"
                >
                    <div className="space-y-1">
                        <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
                            Fixa Control Center
                        </h1>
                        <p className="text-muted-foreground text-sm max-w-md">
                            Orchestrate multi-agent voice testing scenarios with real-time feedback.
                        </p>
                    </div>
                    <ThemeToggle />
                </motion.div>

                <Separator />

                <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 h-[calc(100vh-200px)] min-h-[800px]">
                    {/* Left Column: Configuration (4 cols) */}
                    <div className="xl:col-span-4 flex flex-col gap-6 h-full overflow-hidden">

                        <div className="shrink-0">
                            <TargetAgentSelector />
                        </div>

                        <Separator className="xl:hidden" />

                        <div className="flex-1 min-h-0 flex flex-col gap-6">
                            <Tabs defaultValue="testers" className="flex-1 flex flex-col min-h-0">
                                <TabsList className="grid w-full grid-cols-2">
                                    <TabsTrigger value="testers">Tester Agents</TabsTrigger>
                                    <TabsTrigger value="scenarios">Scenarios</TabsTrigger>
                                </TabsList>
                                <TabsContent value="testers" className="flex-1 min-h-0 mt-4 data-[state=active]:flex flex-col">
                                    <TesterAgentsList />
                                </TabsContent>
                                <TabsContent value="scenarios" className="flex-1 min-h-0 mt-4 data-[state=active]:flex flex-col">
                                    <ScenariosList />
                                </TabsContent>
                            </Tabs>
                        </div>

                        <div className="shrink-0 pt-2">
                             <TestRunnerControl />
                        </div>
                    </div>

                    {/* Right Column: Execution & Results (8 cols) */}
                    <div className="xl:col-span-8 flex flex-col gap-6 h-full min-h-0">
                        <div className="flex-1 min-h-0 relative">
                             <TestResultsView />
                        </div>

                         <div className="h-1/3 min-h-[250px] shrink-0">
                             <LogStreamViewer />
                         </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
