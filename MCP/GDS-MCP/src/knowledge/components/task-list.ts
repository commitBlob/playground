import { ComponentDefinition } from "../types.js";

export const taskList: ComponentDefinition = {
  slug: "task-list",
  name: "Task list",
  category: "content",
  description:
    "Show users a list of tasks they need to complete, with status tags for each task.",
  useCases: [
    "list of tasks to complete",
    "task tracker",
    "application progress",
    "multi-step checklist",
    "show completion status",
  ],
  antiUseCases: [
    "For step-by-step instructions in order, use a numbered list or step-by-step navigation pattern.",
    "If tasks have no status, use a simple list instead.",
  ],
  relatedComponents: ["tag", "summary-list"],
  template: {
    baseMarkup: `<ul class="govuk-task-list">
{{tasks}}
</ul>`,
    slots: [
      { name: "tasks", required: true, type: "array", description: "Tasks with name, href, status, and tag colour" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "task-list-status-not-colour-only",
      severity: "warning",
      description: "Task status should not rely solely on colour",
      check: (html) => {
        if (!/govuk-task-list/.test(html)) return { passed: true, message: "No task list present." };
        return { passed: true, message: "Task list status check requires visual inspection." };
      },
    },
  ],
};
