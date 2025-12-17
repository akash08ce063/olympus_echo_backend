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
import { CheckCircle2, Plus, ScrollText, Trash2 } from 'lucide-react';

export function ScenariosList() {
    const { scenarios, addScenario, removeScenario, updateScenario } = useTestStore();

    const handleAddScenario = () => {
        addScenario({
            id: crypto.randomUUID(),
            name: `Scenario ${scenarios.length + 1}`,
            prompt: '',
            evaluations: []
        });
    };

    const handleAddEvaluation = (scenarioId: string) => {
        const scenario = scenarios.find(s => s.id === scenarioId);
        if (!scenario) return;

        updateScenario(scenarioId, {
            evaluations: [
                ...scenario.evaluations,
                { id: crypto.randomUUID(), name: '', prompt: '' }
            ]
        });
    };

    const handleUpdateEvaluation = (scenarioId: string, evalId: string, field: 'name' | 'prompt', value: string) => {
        const scenario = scenarios.find(s => s.id === scenarioId);
        if (!scenario) return;

        const updatedEvals = scenario.evaluations.map(e =>
            e.id === evalId ? { ...e, [field]: value } : e
        );
        updateScenario(scenarioId, { evaluations: updatedEvals });
    };

    const handleRemoveEvaluation = (scenarioId: string, evalId: string) => {
        const scenario = scenarios.find(s => s.id === scenarioId);
        if (!scenario) return;

        const updatedEvals = scenario.evaluations.filter(e => e.id !== evalId);
        updateScenario(scenarioId, { evaluations: updatedEvals });
    };

    return (
        <Card className="h-full flex flex-col shadow-sm">
             <CardHeader className="flex flex-row items-center justify-between pb-2 bg-muted/20 border-b">
                <div className="space-y-1">
                    <CardTitle className="text-lg flex items-center gap-2">
                         <ScrollText className="w-5 h-5 text-primary" />
                        Test Scenarios
                    </CardTitle>
                    <CardDescription>What situations should be tested?</CardDescription>
                </div>
                <Button size="sm" onClick={handleAddScenario} className="gap-1 shadow-sm">
                    <Plus className="w-4 h-4" /> Add
                </Button>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
                {scenarios.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-40 text-muted-foreground border-2 border-dashed rounded-lg bg-muted/10">
                        <ScrollText className="w-8 h-8 opacity-20 mb-2" />
                        <p className="text-sm">No scenarios defined.</p>
                        <Button variant="link" onClick={handleAddScenario} className="h-auto p-0 text-xs">Create one now</Button>
                    </div>
                ) : (
                     <Accordion type="multiple" defaultValue={scenarios.map(s => s.id)} className="w-full space-y-4">
                        <AnimatePresence>
                             {scenarios.map((scenario) => (
                                <motion.div
                                    key={scenario.id}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.2 }}
                                >
                                     <AccordionItem value={scenario.id} className="border rounded-lg bg-card px-4 shadow-sm">
                                         <div className="flex items-center justify-between py-2">
                                             <AccordionTrigger className="hover:no-underline py-2 flex-1">
                                                 <div className="flex items-center gap-3">
                                                     <span className="font-semibold text-sm">{scenario.name || "Unnamed Scenario"}</span>
                                                     <Badge variant="secondary" className="text-[10px] font-normal">{scenario.evaluations.length} Evals</Badge>
                                                 </div>
                                             </AccordionTrigger>
                                             <Button
                                                 variant="ghost"
                                                 size="icon"
                                                 className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10 -mr-2 ml-2"
                                                 onClick={(e) => {
                                                     e.stopPropagation();
                                                     removeScenario(scenario.id);
                                                 }}
                                             >
                                                 <Trash2 className="w-4 h-4" />
                                             </Button>
                                         </div>

                                        <AccordionContent className="pt-2 pb-4 space-y-4 border-t">
                                            <div className="space-y-2">
                                                <Label className="text-xs">Name</Label>
                                                <Input
                                                    value={scenario.name}
                                                    onChange={(e) => updateScenario(scenario.id, { name: e.target.value })}
                                                    placeholder="e.g. Order Pizza"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs">Instruction Prompt (What the tester should do)</Label>
                                                <Textarea
                                                    value={scenario.prompt}
                                                    onChange={(e) => updateScenario(scenario.id, { prompt: e.target.value })}
                                                    placeholder="Call and order a pepperoni pizza with extra cheese..."
                                                    className="min-h-[80px] text-sm"
                                                />
                                            </div>

                                            <div className="rounded-md bg-muted/40 p-3 space-y-3">
                                                <div className="flex items-center justify-between">
                                                     <Label className="text-xs font-semibold uppercase text-muted-foreground flex items-center gap-1">
                                                        <CheckCircle2 className="w-3 h-3" /> Evaluation Criteria
                                                     </Label>
                                                     <Button size="sm" variant="ghost" className="h-6 text-xs hover:bg-background" onClick={() => handleAddEvaluation(scenario.id)}>
                                                         <Plus className="w-3 h-3 mr-1"/> Add Eval
                                                     </Button>
                                                </div>

                                                <div className="space-y-2">
                                                    {scenario.evaluations.map((evalItem) => (
                                                        <div key={evalItem.id} className="flex gap-2 items-start group">
                                                            <div className="grid grid-cols-3 gap-2 flex-1">
                                                                <Input
                                                                    className="h-8 text-xs bg-background col-span-1"
                                                                    placeholder="Name (e.g. Price Quoted)"
                                                                    value={evalItem.name}
                                                                    onChange={(e) => handleUpdateEvaluation(scenario.id, evalItem.id, 'name', e.target.value)}
                                                                />
                                                                <Input
                                                                    className="h-8 text-xs bg-background col-span-2"
                                                                    placeholder="Success condition (e.g. Agent quotes $15)"
                                                                    value={evalItem.prompt}
                                                                    onChange={(e) => handleUpdateEvaluation(scenario.id, evalItem.id, 'prompt', e.target.value)}
                                                                />
                                                            </div>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-8 w-8 text-muted-foreground hover:text-destructive shrink-0 opacity-50 group-hover:opacity-100 transition-opacity"
                                                                onClick={() => handleRemoveEvaluation(scenario.id, evalItem.id)}
                                                            >
                                                                <Trash2 className="w-3 h-3" />
                                                            </Button>
                                                        </div>
                                                    ))}
                                                    {scenario.evaluations.length === 0 && (
                                                        <div className="text-xs text-muted-foreground italic text-center py-2">
                                                            No automatic evaluations defined.
                                                        </div>
                                                    )}
                                                </div>
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
