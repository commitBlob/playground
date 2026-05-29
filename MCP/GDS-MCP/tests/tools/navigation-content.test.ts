import { describe, it, expect } from "vitest";
import { handleGenerateMarkup } from "../../src/tools/generate-markup.js";
import { handleSuggestComponent } from "../../src/tools/suggest-component.js";
import { handleReviewHtml } from "../../src/tools/review-html.js";

describe("accordion", () => {
  it("generates accordion with sections", () => {
    const result = handleGenerateMarkup({
      component: "accordion",
      options: {
        id: "faq",
        items: [
          { text: "Section 1", value: "Content for section 1" },
          { text: "Section 2", value: "Content for section 2" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-accordion");
    expect(html).toContain('data-module="govuk-accordion"');
    expect(html).toContain('id="faq"');
    expect(html).toContain("Section 1");
    expect(html).toContain("Content for section 1");
  });

  it("is suggested for collapsible content", () => {
    const result = handleSuggestComponent({ useCase: "show and hide sections of content" });
    expect(result.content[0].text).toContain("Accordion");
  });
});

describe("back-link", () => {
  it("generates back link", () => {
    const result = handleGenerateMarkup({
      component: "back-link",
      options: { href: "/previous-page" },
    });
    expect(result.content[0].text).toContain('href="/previous-page"');
    expect(result.content[0].text).toContain("govuk-back-link");
    expect(result.content[0].text).toContain("Back");
  });
});

describe("breadcrumbs", () => {
  it("generates breadcrumbs with ordered list", () => {
    const result = handleGenerateMarkup({
      component: "breadcrumbs",
      options: {
        items: [
          { text: "Home", value: "/" },
          { text: "Services", value: "/services" },
          { text: "Current page", value: "" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain('aria-label="Breadcrumb"');
    expect(html).toContain("<ol");
    expect(html).toContain("govuk-breadcrumbs__link");
    expect(html).toContain('aria-current="page"');
  });

  it("review flags breadcrumbs without aria-label", () => {
    const html = `<nav class="govuk-breadcrumbs"><ol><li>Home</li></ol></nav>`;
    const result = handleReviewHtml({ html, component: "breadcrumbs" });
    expect(result.content[0].text).toContain("aria-label");
  });
});

describe("pagination", () => {
  it("generates previous/next pagination", () => {
    const result = handleGenerateMarkup({
      component: "pagination",
      options: { prevHref: "/page/1", nextHref: "/page/3" },
    });
    const html = result.content[0].text;
    expect(html).toContain('aria-label="Pagination"');
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
    expect(html).toContain("Previous");
    expect(html).toContain("Next");
  });
});

describe("skip-link", () => {
  it("generates skip link", () => {
    const result = handleGenerateMarkup({ component: "skip-link", options: {} });
    const html = result.content[0].text;
    expect(html).toContain("govuk-skip-link");
    expect(html).toContain("#main-content");
    expect(html).toContain("Skip to main content");
  });
});

describe("tabs", () => {
  it("generates tabs with panels", () => {
    const result = handleGenerateMarkup({
      component: "tabs",
      options: {
        items: [
          { text: "Past day", value: "Data for past day" },
          { text: "Past week", value: "Data for past week" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-tabs");
    expect(html).toContain('data-module="govuk-tabs"');
    expect(html).toContain("govuk-tabs__title");
    expect(html).toContain("Past day");
    expect(html).toContain("govuk-tabs__panel");
  });
});

describe("details", () => {
  it("generates details disclosure", () => {
    const result = handleGenerateMarkup({
      component: "details",
      options: { label: "Help with nationality", content: "We need to know your nationality..." },
    });
    const html = result.content[0].text;
    expect(html).toContain("<details");
    expect(html).toContain("govuk-details");
    expect(html).toContain("Help with nationality");
    expect(html).toContain("We need to know");
  });
});

describe("inset-text", () => {
  it("generates inset text block", () => {
    const result = handleGenerateMarkup({
      component: "inset-text",
      options: { content: "It can take up to 8 weeks to register." },
    });
    expect(result.content[0].text).toContain("govuk-inset-text");
    expect(result.content[0].text).toContain("8 weeks");
  });
});

describe("summary-list", () => {
  it("generates summary list with change actions", () => {
    const result = handleGenerateMarkup({
      component: "summary-list",
      options: {
        items: [
          { text: "Name", value: "Sarah Philips", hint: "/change-name" },
          { text: "Date of birth", value: "5 January 1978", hint: "/change-dob" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-summary-list");
    expect(html).toContain("govuk-summary-list__key");
    expect(html).toContain("Sarah Philips");
    expect(html).toContain("govuk-visually-hidden");
    expect(html).toContain("Change");
  });
});

describe("table", () => {
  it("generates table with headers and scope", () => {
    const result = handleGenerateMarkup({
      component: "table",
      options: { label: "Dates and amounts" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-table");
    expect(html).toContain("<caption");
    expect(html).toContain('scope="col"');
  });
});

describe("tag", () => {
  it("generates default tag", () => {
    const result = handleGenerateMarkup({
      component: "tag",
      options: { label: "Completed" },
    });
    expect(result.content[0].text).toContain("govuk-tag");
    expect(result.content[0].text).toContain("Completed");
  });

  it("generates coloured tag", () => {
    const result = handleGenerateMarkup({
      component: "tag",
      options: { label: "In progress", colour: "blue" },
    });
    expect(result.content[0].text).toContain("govuk-tag--blue");
  });
});

describe("error-summary", () => {
  it("generates error summary with links", () => {
    const result = handleGenerateMarkup({
      component: "error-summary",
      options: {
        items: [
          { text: "Enter your full name", value: "full-name" },
          { text: "Enter a valid email", value: "email" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-error-summary");
    expect(html).toContain('role="alert"');
    expect(html).toContain('href="#full-name"');
    expect(html).toContain("There is a problem");
  });
});

describe("notification-banner", () => {
  it("generates standard notification banner", () => {
    const result = handleGenerateMarkup({
      component: "notification-banner",
      options: { label: "This service will be down for maintenance on Friday." },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-notification-banner");
    expect(html).toContain('role="region"');
    expect(html).toContain("Important");
  });

  it("generates success banner", () => {
    const result = handleGenerateMarkup({
      component: "notification-banner",
      options: { label: "Training course updated", variant: "success" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-notification-banner--success");
    expect(html).toContain('role="alert"');
    expect(html).toContain("Success");
  });
});

describe("panel", () => {
  it("generates confirmation panel", () => {
    const result = handleGenerateMarkup({
      component: "panel",
      options: { label: "Application complete", content: "Your reference number: HDJ2123F" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-panel--confirmation");
    expect(html).toContain("Application complete");
    expect(html).toContain("HDJ2123F");
  });
});

describe("warning-text", () => {
  it("generates warning text with hidden prefix", () => {
    const result = handleGenerateMarkup({
      component: "warning-text",
      options: { label: "You can be fined up to £5,000" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-warning-text");
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain("govuk-visually-hidden");
    expect(html).toContain("Warning");
    expect(html).toContain("£5,000");
  });
});
