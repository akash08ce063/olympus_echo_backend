'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { useTestStore } from '@/src/store/useTestStore';
import { Bot } from 'lucide-react';
import { useEffect, useState } from 'react';

export function TargetAgentSelector() {
    const { agents, selectedAgentId, setSelectedAgentId, setAgents } = useTestStore();
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        fetch('/api/agents')
            .then(res => res.json())
            .then(data => {
                if (!data.error) {
                    setAgents(data);
                }
            })
            .catch(err => console.error('Failed to fetch agents:', err))
            .finally(() => setIsLoading(false));
    }, [setAgents]);

    return (
        <Card className="border-l-4 border-l-primary shadow-sm hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                    <Bot className="w-5 h-5 text-primary" />
                    Target Agent
                </CardTitle>
                <CardDescription>Select the agent you want to test</CardDescription>
            </CardHeader>
            <CardContent>
                {isLoading ? (
                    <div className="space-y-2">
                        <Skeleton className="h-10 w-full" />
                        <Skeleton className="h-4 w-2/3" />
                    </div>
                ) : (
                    <div className="space-y-2">
                        <Select value={selectedAgentId || ''} onValueChange={setSelectedAgentId}>
                            <SelectTrigger className="w-full py-6">
                                <SelectValue placeholder="Select an agent..." />
                            </SelectTrigger>
                            <SelectContent>
                                {agents.map((agent) => (
                                    <SelectItem key={agent.id} value={agent.id} className="cursor-pointer">
                                        <div className="flex flex-col text-left">
                                            <span className="font-medium">{agent.name}</span>
                                            <span className="text-xs text-muted-foreground">{agent.phoneNumber}</span>
                                        </div>
                                    </SelectItem>
                                ))}
                                {agents.length === 0 && (
                                    <div className="p-2 text-sm text-center text-muted-foreground">
                                        No agents found
                                    </div>
                                )}
                            </SelectContent>
                        </Select>
                        {selectedAgentId && (
                            <div className="text-xs text-muted-foreground flex justify-between animate-in fade-in slide-in-from-top-1">
                                <span>Selected ID:</span>
                                <span className="font-mono">{selectedAgentId}</span>
                            </div>
                        )}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
