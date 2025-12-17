'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useTestStore } from '@/src/store/useTestStore';
import { Terminal } from 'lucide-react';
import { useEffect, useRef } from 'react';

export function LogStreamViewer() {
    const { logs, addLog, isLoading } = useTestStore();
    const endRef = useRef<HTMLDivElement>(null);

    // Connect to log stream only when test is running
    useEffect(() => {
        // Only connect if a test is currently running
        if (!isLoading) {
            return;
        }

        let eventSource: EventSource | null = null;

        const connectLogs = () => {
            // Close existing if open (though useEffect cleanup handles this)
            if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                eventSource.close();
            }

            eventSource = new EventSource('/api/logs');

            eventSource.onopen = () => {
                console.log('Connected to log stream');
            };

            eventSource.onmessage = (event) => {
                try {
                    const raw = event.data;
                    addLog(raw);
                } catch (e) {
                    console.error('Error parsing log:', e);
                }
            };

            eventSource.onerror = (err) => {
                console.error('Log stream error:', err);
                eventSource?.close();
            };
        };

        connectLogs();

        return () => {
            eventSource?.close();
        };
    }, [addLog, isLoading]);

    // Auto-scroll to bottom
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <Card className="flex flex-col h-full bg-zinc-950 border-zinc-800 shadow-inner">
            <CardHeader className="py-3 px-4 border-b border-zinc-800 bg-zinc-900/50">
                <div className="flex items-center justify-between">
                     <CardTitle className="text-sm font-mono text-zinc-400 flex items-center gap-2">
                        <Terminal className="w-4 h-4" />
                        System Logs
                    </CardTitle>
                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-500 bg-zinc-900">
                        {logs.length} Lines
                    </Badge>
                </div>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-[200px] h-[300px]">
                <ScrollArea className="h-full w-full p-4">
                    <div className="font-mono text-xs text-zinc-300 space-y-1">
                        {logs.length === 0 && (
                            <div className="text-zinc-600 italic">Waiting for logs...</div>
                        )}
                        {logs.map((log, i) => (
                            <div key={i} className="whitespace-pre-wrap break-all border-b border-zinc-900/50 pb-0.5 mb-0.5 last:border-0">
                                <span className="text-zinc-600 mr-2 select-none">{i + 1}</span>
                                {log}
                            </div>
                        ))}
                         <div ref={endRef} />
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
