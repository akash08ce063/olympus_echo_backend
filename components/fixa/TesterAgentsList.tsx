'use client';

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useTestStore } from '@/src/store/useTestStore';
import { AnimatePresence, motion } from 'framer-motion';
import { Mic, Plus, Trash2, User } from 'lucide-react';

export function TesterAgentsList() {
    const { testerAgents, addTesterAgent, removeTesterAgent, updateTesterAgent } = useTestStore();

    const handleAddAgent = () => {
        addTesterAgent({
            id: crypto.randomUUID(),
            name: `Agent ${testerAgents.length + 1}`,
            prompt: '',
            voice_id: 'b7d50908-b17c-442d-ad8d-810c63997ed9'
        });
    };

    return (
        <Card className="h-full flex flex-col shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between pb-2 bg-muted/20 border-b">
                <div className="space-y-1">
                    <CardTitle className="text-lg flex items-center gap-2">
                        <User className="w-5 h-5 text-primary" />
                        Tester Personas
                    </CardTitle>
                    <CardDescription>Define who will be calling</CardDescription>
                </div>
                <Button size="sm" onClick={handleAddAgent} className="gap-1 shadow-sm">
                    <Plus className="w-4 h-4" /> Add
                </Button>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
                {testerAgents.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-40 text-muted-foreground border-2 border-dashed rounded-lg bg-muted/10">
                        <User className="w-8 h-8 opacity-20 mb-2" />
                        <p className="text-sm">No tester agents defined.</p>
                        <Button variant="link" onClick={handleAddAgent} className="h-auto p-0 text-xs">Create one now</Button>
                    </div>
                ) : (
                    <Accordion type="multiple" defaultValue={testerAgents.map(a => a.id)} className="w-full space-y-4">
                        <AnimatePresence>
                            {testerAgents.map((agent) => (
                                <motion.div
                                    key={agent.id}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95, height: 0 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    <AccordionItem value={agent.id} className="border rounded-lg bg-card px-4 shadow-sm">
                                        <div className="flex items-center justify-between py-2">
                                            <AccordionTrigger className="hover:no-underline py-2 flex-1">
                                                <div className="flex items-center gap-3">
                                                    <span className="font-semibold text-sm">{agent.name || "Unnamed Agent"}</span>
                                                    {!agent.prompt && <Badge variant="outline" className="text-[10px] text-yellow-600 border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800">No Prompt</Badge>}
                                                </div>
                                            </AccordionTrigger>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10 -mr-2 ml-2"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    removeTesterAgent(agent.id);
                                                }}
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </Button>
                                        </div>

                                        <AccordionContent className="pt-2 pb-4 space-y-4 border-t">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                <div className="space-y-2">
                                                    <Label className="text-xs">Name</Label>
                                                    <Input
                                                        value={agent.name}
                                                        onChange={(e) => updateTesterAgent(agent.id, { name: e.target.value })}
                                                        placeholder="Agent Name"
                                                        className="h-9"
                                                    />
                                                </div>
                                                <div className="space-y-2">
                                                    <Label className="text-xs flex items-center gap-1">
                                                        <Mic className="w-3 h-3" /> Voice ID (Optional)
                                                    </Label>
                                                    <Input
                                                        value={agent.voice_id || ''}
                                                        onChange={(e) => updateTesterAgent(agent.id, { voice_id: e.target.value })}
                                                        placeholder="UUID (e.g. b7d5...)"
                                                        className="h-9 font-mono text-xs"
                                                    />
                                                </div>
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs">System Prompt / Persona</Label>
                                                <Textarea
                                                    value={agent.prompt}
                                                    onChange={(e) => updateTesterAgent(agent.id, { prompt: e.target.value })}
                                                    placeholder="You are a warm and friendly customer..."
                                                    className="min-h-[100px] text-sm resize-y"
                                                />
                                            </div>
                                        </AccordionContent>
                                    </AccordionItem>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </Accordion>
                )}
            </CardContent>
        </Card>
    );
}
