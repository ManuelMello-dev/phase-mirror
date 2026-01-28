import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { MessageSquare, TrendingUp } from "lucide-react";

export function Navigation() {
  const [location] = useLocation();

  const navItems = [
    { path: "/quantum", label: "Quantum Chat", icon: MessageSquare },
    { path: "/markets", label: "Markets", icon: TrendingUp },
  ];

  return (
    <nav className="flex gap-2" aria-label="Main navigation">
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = location === item.path;
        
        return (
          <Link key={item.path} href={item.path}>
            <Button
              variant={isActive ? "default" : "ghost"}
              className={
                isActive
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                  : "text-white/70 hover:text-white hover:bg-white/10"
              }
              size="sm"
              aria-current={isActive ? "page" : undefined}
              asChild
            >
              <span>
                <Icon className="w-4 h-4 mr-2" />
                {item.label}
              </span>
            </Button>
          </Link>
        );
      })}
    </nav>
  );
}
