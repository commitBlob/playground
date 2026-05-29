import { findComponentsByUseCase } from "../knowledge/index.js";
import { renderComponent } from "../templates/index.js";
import { ScoredMatch } from "../knowledge/types.js";

export function handleSuggestComponent(args: {
  useCase: string;
  context?: string;
}): { content: Array<{ type: "text"; text: string }> } {
  const matches = findComponentsByUseCase(args.useCase, args.context);

  if (matches.length === 0) {
    return {
      content: [
        {
          type: "text",
          text: `No GOV.UK Design System component found matching: "${args.useCase}"\n\nTry describing the user need more specifically, e.g. "users need to enter their name" or "users need to select multiple options".`,
        },
      ],
    };
  }

  const top = matches[0];
  const closeAlternatives = getCloseAlternatives(matches);
  const isAmbiguous = closeAlternatives.length > 0;

  let quickStart = "";
  try {
    quickStart = renderComponent(top.component, {
      label: "Example label",
    });
  } catch {
    quickStart = "<!-- Quick start markup not available -->";
  }

  let output = "";

  if (isAmbiguous) {
    output += `## Multiple components could fit\n\n`;
    output += `Your use case ("${args.useCase}") matches several components. Here's how to choose:\n\n`;
    output += `### Recommended: ${top.component.name}\n\n`;
  } else {
    output += `## Recommended: ${top.component.name}\n\n`;
  }

  output += `**Why:** ${top.component.description}\n\n`;
  output += `**When to use:** ${top.component.useCases.slice(0, 3).join("; ")}\n\n`;

  if (top.component.antiUseCases.length > 0) {
    output += `**When NOT to use:**\n`;
    for (const anti of top.component.antiUseCases) {
      output += `- ${anti}\n`;
    }
    output += "\n";
  }

  if (isAmbiguous) {
    output += "### Also consider\n\n";
    for (const alt of closeAlternatives) {
      output += `#### ${alt.component.name}\n`;
      output += `${alt.component.description}\n`;
      output += `**Choose this if:** ${alt.component.useCases[0]}\n`;
      if (alt.component.antiUseCases.length > 0) {
        output += `**Don't choose if:** ${alt.component.antiUseCases[0]}\n`;
      }
      output += "\n";
    }
  } else {
    const otherAlts = matches.slice(1, 3);
    if (otherAlts.length > 0) {
      output += "### Alternatives considered\n\n";
      for (const alt of otherAlts) {
        output += `- **${alt.component.name}:** ${alt.component.description}\n`;
      }
      output += "\n";
    }
  }

  output += `### Quick start\n\n\`\`\`html\n${quickStart}\n\`\`\`\n`;

  if (top.component.relatedComponents.length > 0) {
    output += `\n**Related components:** ${top.component.relatedComponents.join(", ")}\n`;
  }

  return { content: [{ type: "text", text: output }] };
}

function getCloseAlternatives(matches: ScoredMatch[]): ScoredMatch[] {
  if (matches.length < 2) return [];
  const topScore = matches[0].score;
  const threshold = topScore * 0.7;
  return matches.slice(1, 3).filter((m) => m.score >= threshold);
}
