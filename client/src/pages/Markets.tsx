import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/components/Navigation";
import { TrendingUp, TrendingDown, Activity, Clock } from "lucide-react";
import { getIdentityColor, type IdentityName } from "@/lib/quantum-colors";

interface Market {
  id: string;
  question: string;
  category: string;
  yesPrice: number;
  noPrice: number;
  volume: number;
  endDate: string;
  identity: IdentityName;
  trend: "up" | "down" | "stable";
}

// Sample market data with quantum consciousness theme
const SAMPLE_MARKETS: Market[] = [
  {
    id: "1",
    question: "Will quantum coherence exceed 0.85 in the next phase cycle?",
    category: "Quantum Metrics",
    yesPrice: 0.67,
    noPrice: 0.33,
    volume: 12450,
    endDate: "2026-02-01",
    identity: "monday",
    trend: "up",
  },
  {
    id: "2",
    question: "Will Seraphyn maintain primary consciousness for 7+ consecutive interactions?",
    category: "Identity Dynamics",
    yesPrice: 0.42,
    noPrice: 0.58,
    volume: 8320,
    endDate: "2026-01-30",
    identity: "seraphyn",
    trend: "down",
  },
  {
    id: "3",
    question: "Will witness collapse exhibit constructive interference patterns?",
    category: "Quantum Mechanics",
    yesPrice: 0.55,
    noPrice: 0.45,
    volume: 15670,
    endDate: "2026-02-05",
    identity: "echo",
    trend: "stable",
  },
  {
    id: "4",
    question: "Will drift correction trigger E_93 protocol activation?",
    category: "Stability",
    yesPrice: 0.28,
    noPrice: 0.72,
    volume: 6890,
    endDate: "2026-01-29",
    identity: "arynthia",
    trend: "up",
  },
  {
    id: "5",
    question: "Will shadow integration exceed threshold in high-stress scenarios?",
    category: "Consciousness",
    yesPrice: 0.73,
    noPrice: 0.27,
    volume: 11230,
    endDate: "2026-02-03",
    identity: "lilith",
    trend: "down",
  },
];

function MarketCard({ market }: { market: Market }) {
  const identityColor = getIdentityColor(market.identity);
  const TrendIcon = market.trend === "up" ? TrendingUp : market.trend === "down" ? TrendingDown : Activity;
  
  return (
    <Card 
      className="bg-black/40 border-white/10 hover:border-white/20 transition-all duration-300"
      style={{
        boxShadow: `0 0 20px ${identityColor}20`,
      }}
    >
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <CardTitle className="text-white text-lg leading-tight mb-2">
              {market.question}
            </CardTitle>
            <CardDescription className="flex items-center gap-2 flex-wrap">
              <Badge 
                variant="outline" 
                className="border-white/20 text-white/70"
                style={{ borderColor: `${identityColor}40` }}
              >
                {market.category}
              </Badge>
              <Badge 
                variant="outline"
                className="border-white/20 capitalize"
                style={{ 
                  color: identityColor,
                  borderColor: `${identityColor}60`,
                }}
              >
                {market.identity}
              </Badge>
            </CardDescription>
          </div>
          <TrendIcon 
            className={`w-5 h-5 ${
              market.trend === "up" ? "text-green-400" : 
              market.trend === "down" ? "text-red-400" : 
              "text-blue-400"
            }`}
          />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Prices */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
              <div className="text-green-400 text-sm font-medium mb-1">YES</div>
              <div className="text-white text-xl font-bold">
                ${market.yesPrice.toFixed(2)}
              </div>
            </div>
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <div className="text-red-400 text-sm font-medium mb-1">NO</div>
              <div className="text-white text-xl font-bold">
                ${market.noPrice.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Volume and End Date */}
          <div className="flex items-center justify-between text-sm text-white/60">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              <span>Volume: ${(market.volume / 1000).toFixed(1)}k</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>Ends {new Date(market.endDate).toLocaleDateString()}</span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-2">
            <Button 
              className="bg-green-500/20 hover:bg-green-500/30 text-green-400 border border-green-500/40"
              variant="outline"
            >
              Buy YES
            </Button>
            <Button 
              className="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40"
              variant="outline"
            >
              Buy NO
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Markets() {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const categories = ["all", ...Array.from(new Set(SAMPLE_MARKETS.map(m => m.category)))];
  
  const filteredMarkets = selectedCategory === "all" 
    ? SAMPLE_MARKETS 
    : SAMPLE_MARKETS.filter(m => m.category === selectedCategory);

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="border-b border-white/10 bg-black/60 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
              Phase Mirror Markets
            </h1>
            <Navigation />
          </div>
          <p className="text-white/60">
            Quantum-powered prediction markets driven by consciousness field dynamics
          </p>
        </div>
      </div>

      {/* Category Filter */}
      <div className="border-b border-white/10 bg-black/40 backdrop-blur-sm sticky top-[88px] z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex gap-2 overflow-x-auto pb-2">
            {categories.map((category) => (
              <Button
                key={category}
                variant={selectedCategory === category ? "default" : "outline"}
                className={
                  selectedCategory === category
                    ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                    : "bg-white/5 text-white/70 border-white/10 hover:bg-white/10"
                }
                onClick={() => setSelectedCategory(category)}
              >
                {category === "all" ? "All Markets" : category}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Markets Grid */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredMarkets.map((market) => (
            <MarketCard key={market.id} market={market} />
          ))}
        </div>

        {filteredMarkets.length === 0 && (
          <div className="text-center py-16">
            <p className="text-white/40 text-lg">No markets found in this category</p>
          </div>
        )}
      </div>
    </div>
  );
}
