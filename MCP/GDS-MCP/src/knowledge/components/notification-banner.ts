import { ComponentDefinition } from "../types.js";

export const notificationBanner: ComponentDefinition = {
  slug: "notification-banner",
  name: "Notification banner",
  category: "messaging",
  description:
    "A prominent banner for important page-level messages, either as a static notice or a success confirmation.",
  useCases: [
    "important page notice",
    "success message after action",
    "service-wide notification",
    "confirmation banner",
    "alert the user to something",
  ],
  antiUseCases: [
    "Don't overuse — research shows users often miss banners if they appear on every page.",
    "For form errors, use Error summary instead.",
    "For final confirmation pages, use Panel instead.",
  ],
  relatedComponents: ["panel", "error-summary"],
  template: {
    baseMarkup: `<div class="govuk-notification-banner{{successClass}}" role="{{role}}" aria-labelledby="govuk-notification-banner-title" data-module="govuk-notification-banner">
  <div class="govuk-notification-banner__header">
    <h2 class="govuk-notification-banner__title" id="govuk-notification-banner-title">
      {{title}}
    </h2>
  </div>
  <div class="govuk-notification-banner__content">
    <p class="govuk-notification-banner__heading">
      {{heading}}
    </p>
  </div>
</div>`,
    slots: [
      { name: "title", required: false, type: "text", description: "Banner title (default: 'Important')", default: "Important" },
      { name: "heading", required: true, type: "text", description: "Main banner message" },
      { name: "type", required: false, type: "text", description: "'success' for success banners, omit for standard notices" },
    ],
    variants: [
      { name: "success", description: "Green success banner with role=\"alert\"" },
    ],
  },
  accessibilityRules: [
    {
      id: "notification-banner-role",
      severity: "error",
      description: "Notification banners must have the correct role (region for notices, alert for success)",
      check: (html) => {
        if (!/govuk-notification-banner/.test(html)) return { passed: true, message: "No notification banner present." };
        const hasRole = /role=["'](region|alert)["']/.test(html);
        if (!hasRole) {
          return { passed: false, message: "Notification banner missing role attribute.", suggestion: "Add role=\"region\" for standard notices or role=\"alert\" for success messages." };
        }
        return { passed: true, message: "Notification banner has correct role." };
      },
    },
    {
      id: "notification-banner-aria-labelledby",
      severity: "warning",
      description: "Notification banner should be labelled by its title",
      check: (html) => {
        if (!/govuk-notification-banner/.test(html)) return { passed: true, message: "No notification banner present." };
        if (!/aria-labelledby/.test(html)) {
          return { passed: false, message: "Notification banner missing aria-labelledby.", suggestion: "Add aria-labelledby pointing to the banner title id." };
        }
        return { passed: true, message: "Notification banner has aria-labelledby." };
      },
    },
  ],
};
