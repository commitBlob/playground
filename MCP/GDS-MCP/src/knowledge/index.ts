import { ComponentDefinition, ScoredMatch } from "./types.js";
import { getAllComponents, getComponentBySlug } from "./components/index.js";

export { getComponentBySlug } from "./components/index.js";
export { runAllRules, runComponentRules } from "./rules/index.js";
export type { ReviewResult } from "./rules/index.js";

const STOP_WORDS = new Set([
  "the", "that", "this", "with", "from", "they", "them", "their",
  "have", "has", "had", "will", "would", "could", "should", "need",
  "needs", "want", "like", "into", "some", "also", "just", "very",
  "users", "user",
]);

export function findComponentsByUseCase(
  useCase: string,
  context?: string
): ScoredMatch[] {
  const components = getAllComponents();
  const query = `${useCase} ${context || ""}`.toLowerCase();
  const queryWords = query
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));

  const scored: ScoredMatch[] = components.map((component) => {
    let score = 0;
    let reason = "";

    for (const uc of component.useCases) {
      const ucLower = uc.toLowerCase();
      let matchCount = 0;
      let weightedCount = 0;
      for (const word of queryWords) {
        if (ucLower.includes(word)) {
          matchCount++;
          weightedCount += Math.min(word.length / 5, 1.5);
        }
      }
      if (matchCount > 0) {
        const wordScore = weightedCount / queryWords.length;
        if (wordScore > score) {
          score = wordScore;
          reason = uc;
        }
      }
    }

    const descLower = component.description.toLowerCase();
    let descMatches = 0;
    for (const word of queryWords) {
      if (descLower.includes(word)) descMatches++;
    }
    if (descMatches > 0) {
      score += (descMatches / queryWords.length) * 0.3;
    }

    // Negative matching: penalise if antiUseCases explicitly redirect away
    for (const anti of component.antiUseCases) {
      const antiLower = anti.toLowerCase();
      let antiMatchCount = 0;
      for (const word of queryWords) {
        if (antiLower.includes(word)) antiMatchCount++;
      }
      if (antiMatchCount >= 2) {
        score *= 0.5;
      }
    }

    // Boost exact slug match
    if (query.includes(component.slug)) {
      score += 1.5;
    }

    // Boost exact name match
    if (query.includes(component.name.toLowerCase())) {
      score += 1.0;
    }

    return { component, score, reason };
  });

  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score);
}

export function getComponent(slug: string): ComponentDefinition | undefined {
  return getComponentBySlug(slug);
}
