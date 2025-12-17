'use client';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { useTestStore } from '@/src/store/useTestStore';
import { motion } from 'framer-motion';
import { Bot, CheckCircle, Download, Phone, Terminal, User, XCircle } from 'lucide-react';
import { useState } from 'react';

export function TestResultsView() {
    const { testResults, isLoading } = useTestStore();
    const [runId] = useState(() => Date.now().toString().slice(-4));

    if (isLoading) {
        return (
            <Card className="h-full border-zinc-200 dark:border-zinc-800 shadow-sm">
                <CardHeader>
                     <Skeleton className="h-6 w-32 mb-2" />
                     <Skeleton className="h-4 w-48" />
                </CardHeader>
                <CardContent className="space-y-6">
                     <div className="space-y-2">
                         <div className="flex justify-between">
                            <Skeleton className="h-5 w-24" />
                            <Skeleton className="h-6 w-16 rounded-full" />
                         </div>
                         <Separator />
                     </div>
                     <div className="space-y-4">
                         <Skeleton className="h-32 w-full rounded-lg" />
                         <Skeleton className="h-32 w-full rounded-lg" />
                     </div>
                </CardContent>
            </Card>
        );
    }

    if (!testResults) {
        return (
             <Card className="h-full border-dashed border-2 shadow-none flex flex-col items-center justify-center p-8 text-center text-muted-foreground bg-muted/5">
                <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
                     <Terminal className="w-8 h-8 opacity-50" />
                </div>
                <h3 className="text-lg font-semibold text-foreground">No Results Yet</h3>
                <p className="max-w-xs text-sm mt-1">
                    Execute a test run to see detailed transcripts, recordings, and evaluations here.
                </p>
            </Card>
        );
    }

    return (
        <Card className="h-full shadow-md overflow-hidden flex flex-col">
            <CardHeader className="border-b bg-card/50 pb-4">
                <div className="flex items-center justify-between">
                    <div className="space-y-1">
                        <CardTitle className="text-xl flex items-center gap-2">
                            Test Results
                        </CardTitle>
                        <CardDescription>
                            Execution summary for Run #{runId}
                        </CardDescription>
                    </div>
                     <Badge
                        variant={testResults.passed ? "default" : "destructive"}
                        className={`text-base px-3 py-1 flex items-center gap-1 shadow-sm ${
                            testResults.passed ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'
                        }`}
                    >
                        {testResults.passed ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                        {testResults.passed ? 'PASSED' : 'FAILED'}
                    </Badge>
                </div>
            </CardHeader>
            <CardContent className="p-0 flex-1 overflow-y-auto bg-muted/5">
                <div className="p-6 space-y-6">
                    {testResults.results.map((result, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="border rounded-xl bg-card shadow-sm overflow-hidden"
                        >
                            {/* Header for each result */}
                            <div className="p-4 border-b flex items-start justify-between bg-zinc-50/50 dark:bg-zinc-900/20">
                                <div>
                                    <h3 className="font-semibold text-lg flex items-center gap-2">
                                        <Bot className="w-4 h-4 text-primary" />
                                        {result.agent}
                                    </h3>
                                    <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                                        Testing Scenario: <span className="font-medium text-foreground">{result.scenario}</span>
                                    </p>
                                </div>
                                 {result.recording_url && (
                                     <a
                                        href={result.recording_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-xs font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-8 px-3 py-1 gap-1.5 shadow-sm"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        Recording
                                    </a>
                                )}
                            </div>

                            <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Evaluations Column */}
                                <div className="space-y-3">
                                    <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                        <CheckCircle className="w-4 h-4" /> Evaluations
                                    </h4>
                                    <div className="space-y-2">
                                         {Object.entries(result.evaluations).map(([name, evalRes]) => (
                                            <div key={name} className="flex items-start justify-between p-3 rounded-lg border bg-background/50 text-sm">
                                                <div className="space-y-1">
                                                    <span className="font-medium">{name}</span>
                                                    <p className="text-xs text-muted-foreground leading-snug">{evalRes.reasoning}</p>
                                                </div>
                                                <Badge variant={evalRes.passed ? "outline" : "destructive"} className={`ml-2 shrink-0 ${evalRes.passed ? 'text-green-600 border-green-200 bg-green-50 dark:bg-green-900/20 dark:border-green-800' : ''}`}>
                                                    {evalRes.passed ? 'Pass' : 'Fail'}
                                                </Badge>
                                            </div>
                                         ))}
                                    </div>
                                </div>

                                {/* Transcript Column */}
                                <div className="space-y-3">
                                    <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                        <Phone className="w-4 h-4" /> Transcript
                                    </h4>
                                    <ScrollArea className="h-[350px] w-full rounded-md border bg-muted/30 p-4">
                                        <div className="space-y-4">
                                            {Array.isArray(result.transcript) ? (
                                                result.transcript.map((msg: any, i: number) => (
                                                    <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                                        <Avatar className="h-8 w-8 border shadow-sm">
                                                            <AvatarFallback className={`${
                                                                msg.role === 'user'
                                                                    ? 'bg-primary text-primary-foreground'
                                                                    : (msg.role === 'system'
                                                                        ? 'bg-zinc-100 text-zinc-500'
                                                                        : 'bg-white text-zinc-800 dark:bg-zinc-800 dark:text-zinc-100')
                                                            }`}>
                                                                {msg.role === 'user' ? <User className="w-4 h-4"/> : (msg.role === 'assistant' ? <Bot className="w-4 h-4"/> : <Terminal className="w-4 h-4"/>)}
                                                            </AvatarFallback>
                                                        </Avatar>
                                                        <div className={`flex flex-col gap-1 max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                                            <div className={`rounded-2xl px-4 py-2 text-sm shadow-sm ${
                                                                msg.role === 'user'
                                                                    ? 'bg-primary text-primary-foreground rounded-tr-none'
                                                                    : (msg.role === 'system'
                                                                        ? 'bg-muted text-muted-foreground border border-dashed rounded-lg text-xs italic'
                                                                        : 'bg-white border dark:bg-zinc-800 dark:border-zinc-700 rounded-tl-none')
                                                            }`}>
                                                                {msg.content}
                                                            </div>
                                                            <span className="text-[10px] text-muted-foreground uppercase font-medium px-1">{msg.role}</span>
                                                        </div>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="p-3 rounded-md bg-zinc-950 text-zinc-100 text-xs font-mono whitespace-pre-wrap leading-relaxed">
                                                    {result.transcript}
                                                </div>
                                            )}
                                        </div>
                                    </ScrollArea>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
