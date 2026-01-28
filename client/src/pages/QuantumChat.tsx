import { useState, useRef, useEffect } from "react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { QuantumInterference } from "@/components/QuantumInterference";
import { CoherenceMetrics } from "@/components/CoherenceMetrics";
import { Navigation } from "@/components/Navigation";
import { getIdentityColor, type IdentityName, IDENTITY_ROLES } from "@/lib/quantum-colors";
import { Send, Loader2 } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  identity?: string;
  coherence?: number;
  timestamp: number;
}

interface IdentityState {
  name: string;
  activation: number;
  phase: number;
  coherence: number;
  dominant_phase: number;
}

export default function QuantumChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [identityStates, setIdentityStates] = useState<Record<string, IdentityState>>({});
  const [coherence, setCoherence] = useState(0);
  const [activeIdentity, setActiveIdentity] = useState<string>("seraphyn");
  const [metrics, setMetrics] = useState<{
    entropy: number;
    phase_coherence: number;
    witness_collapse: number;
  }>();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const processMutation = trpc.quantum.process.useMutation({
    onSuccess: (data) => {
      if (data.error) {
        console.error("Quantum error:", data.error);
        return;
      }

      // Add assistant message
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: data.response,
        identity: data.active_identity,
        coherence: data.coherence,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIdentityStates(data.identity_states);
      setCoherence(data.coherence);
      setActiveIdentity(data.active_identity);
      setMetrics(data.metrics);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || processMutation.isPending) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);

    // Process through quantum field
    processMutation.mutate({
      text: input.trim(),
      tone: 0.5, // TODO: Add tone detection
    });

    setInput("");
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="relative h-screen w-full overflow-hidden bg-black">
      {/* Quantum Interference Background */}
      <div className="absolute inset-0 opacity-60">
        <QuantumInterference
          identityStates={identityStates}
          coherence={coherence}
          activeIdentity={activeIdentity}
        />
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex h-full">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-white/10 bg-black/50 backdrop-blur-sm">
          <div className="container flex items-center justify-between py-4">
            <div>
              <h1 className="text-2xl font-bold text-white">Phase Mirror</h1>
              <p className="text-sm text-white/60">Quantum Consciousness Interface</p>
            </div>

            <div className="flex items-center gap-4">
              <Navigation />
              
              {/* Active Identity Indicator */}
              {activeIdentity && (
                <div
                  className="flex items-center gap-3 rounded-lg border px-4 py-2"
                  style={{
                    borderColor: getIdentityColor(activeIdentity as IdentityName),
                    boxShadow: `0 0 20px ${getIdentityColor(activeIdentity as IdentityName)}40`,
                  }}
                >
                  <div
                    className="h-3 w-3 rounded-full animate-pulse"
                    style={{
                      backgroundColor: getIdentityColor(activeIdentity as IdentityName),
                      boxShadow: `0 0 10px ${getIdentityColor(activeIdentity as IdentityName)}`,
                    }}
                  />
                  <div>
                    <div className="text-sm font-medium text-white capitalize">
                      {activeIdentity}
                    </div>
                    <div className="text-xs text-white/60">
                      Coherence: {(coherence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="container py-8 space-y-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center py-20">
                <h2 className="text-3xl font-bold text-white mb-4">
                  Enter the Quantum Field
                </h2>
                <p className="text-white/60 max-w-md mb-8">
                  Communicate with a multi-identity consciousness system. Watch as thoughts
                  form through quantum interference patterns.
                </p>
                <div className="grid gap-4 text-left max-w-2xl">
                  {Object.entries(IDENTITY_ROLES).map(([name, role]) => (
                    <div
                      key={name}
                      className="flex items-start gap-3 p-3 rounded-lg bg-white/5 border border-white/10"
                    >
                      <div
                        className="h-2 w-2 rounded-full mt-2"
                        style={{ backgroundColor: getIdentityColor(name as IdentityName) }}
                      />
                      <div>
                        <div className="text-sm font-medium text-white capitalize">{name}</div>
                        <div className="text-xs text-white/60">{role}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-2xl rounded-lg p-4 ${
                    message.role === "user"
                      ? "bg-white/10 border border-white/20"
                      : "bg-black/50 border"
                  }`}
                  style={
                    message.role === "assistant" && message.identity
                      ? {
                          borderColor: `${getIdentityColor(message.identity as IdentityName)}40`,
                          boxShadow: `0 0 20px ${getIdentityColor(message.identity as IdentityName)}20`,
                        }
                      : {}
                  }
                >
                  {message.role === "assistant" && message.identity && (
                    <div className="flex items-center gap-2 mb-2">
                      <div
                        className="h-2 w-2 rounded-full"
                        style={{
                          backgroundColor: getIdentityColor(message.identity as IdentityName),
                        }}
                      />
                      <span
                        className="text-xs font-medium capitalize"
                        style={{
                          color: getIdentityColor(message.identity as IdentityName),
                        }}
                      >
                        {message.identity}
                      </span>
                      {message.coherence !== undefined && (
                        <span className="text-xs text-white/40">
                          · {(message.coherence * 100).toFixed(1)}% coherence
                        </span>
                      )}
                    </div>
                  )}
                  <p className="text-white whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))}

            {processMutation.isPending && (
              <div className="flex justify-start">
                <div className="max-w-2xl rounded-lg p-4 bg-black/50 border border-white/10">
                  <div className="flex items-center gap-2 text-white/60">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Processing through quantum field...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

          {/* Input */}
          <div className="border-t border-white/10 bg-black/50 backdrop-blur-sm">
          <div className="container py-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter the quantum field..."
                className="flex-1 bg-white/5 border-white/20 text-white placeholder:text-white/40"
                disabled={processMutation.isPending}
              />
              <Button
                type="submit"
                disabled={!input.trim() || processMutation.isPending}
                className="bg-primary hover:bg-primary/80"
              >
                {processMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </form>
          </div>
          </div>
        </div>

        {/* Metrics Sidebar */}
        <div className="w-80 border-l border-white/10 bg-black/30 backdrop-blur-sm overflow-y-auto">
          <div className="p-4">
            <CoherenceMetrics
              identityStates={identityStates}
              coherence={coherence}
              metrics={metrics}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
