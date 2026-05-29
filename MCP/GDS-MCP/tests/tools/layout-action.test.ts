import { describe, it, expect } from "vitest";
import { handleGenerateMarkup } from "../../src/tools/generate-markup.js";
import { handleSuggestComponent } from "../../src/tools/suggest-component.js";

describe("button", () => {
  it("generates primary button", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: { label: "Save and continue" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-button");
    expect(html).toContain('data-module="govuk-button"');
    expect(html).toContain("Save and continue");
    expect(html).toContain('type="submit"');
  });

  it("generates secondary button", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: { label: "Find address", variant: "secondary" },
    });
    expect(result.content[0].text).toContain("govuk-button--secondary");
  });

  it("generates warning button", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: { label: "Delete account", variant: "warning" },
    });
    expect(result.content[0].text).toContain("govuk-button--warning");
  });

  it("generates start button with SVG arrow", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: { label: "Start now", variant: "start", href: "/start" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-button--start");
    expect(html).toContain("govuk-button__start-icon");
    expect(html).toContain('role="button"');
    expect(html).toContain('href="/start"');
  });

  it("generates disabled button with aria-disabled", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: { label: "Submit", disabled: true },
    });
    const html = result.content[0].text;
    expect(html).toContain("disabled");
    expect(html).toContain('aria-disabled="true"');
  });

  it("is suggested for form submission", () => {
    const result = handleSuggestComponent({ useCase: "submit a form" });
    expect(result.content[0].text).toContain("Button");
  });
});

describe("header", () => {
  it("generates GOV.UK header with service name", () => {
    const result = handleGenerateMarkup({
      component: "header",
      options: { serviceName: "Apply for a licence" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-header");
    expect(html).toContain("GOV.UK");
    expect(html).toContain("Apply for a licence");
    expect(html).toContain("govuk-header__service-name");
  });

  it("generates header without service name", () => {
    const result = handleGenerateMarkup({ component: "header", options: {} });
    const html = result.content[0].text;
    expect(html).toContain("govuk-header");
    expect(html).toContain("GOV.UK");
    expect(html).not.toContain("govuk-header__service-name");
  });
});

describe("footer", () => {
  it("generates GOV.UK footer", () => {
    const result = handleGenerateMarkup({ component: "footer", options: {} });
    const html = result.content[0].text;
    expect(html).toContain("govuk-footer");
    expect(html).toContain("Crown copyright");
    expect(html).toContain("Open Government Licence");
  });
});

describe("cookie-banner", () => {
  it("generates cookie banner with service name", () => {
    const result = handleGenerateMarkup({
      component: "cookie-banner",
      options: { serviceName: "Apply for a driving licence" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-cookie-banner");
    expect(html).toContain('role="region"');
    expect(html).toContain("Apply for a driving licence");
    expect(html).toContain("Accept analytics cookies");
    expect(html).toContain("Reject analytics cookies");
  });
});

describe("exit-this-page", () => {
  it("generates exit button", () => {
    const result = handleGenerateMarkup({ component: "exit-this-page", options: {} });
    const html = result.content[0].text;
    expect(html).toContain("govuk-exit-this-page");
    expect(html).toContain('role="button"');
    expect(html).toContain("Exit this page");
    expect(html).toContain("govuk-visually-hidden");
  });
});

describe("phase-banner", () => {
  it("generates beta phase banner", () => {
    const result = handleGenerateMarkup({ component: "phase-banner", options: {} });
    const html = result.content[0].text;
    expect(html).toContain("govuk-phase-banner");
    expect(html).toContain("beta");
    expect(html).toContain("feedback");
  });

  it("generates alpha phase banner", () => {
    const result = handleGenerateMarkup({
      component: "phase-banner",
      options: { phase: "alpha" },
    });
    expect(result.content[0].text).toContain("alpha");
  });
});

describe("service-navigation", () => {
  it("generates service nav with links", () => {
    const result = handleGenerateMarkup({
      component: "service-navigation",
      options: {
        serviceName: "My service",
        items: [
          { text: "Dashboard", value: "/dashboard" },
          { text: "Settings", value: "/settings" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-service-navigation");
    expect(html).toContain("My service");
    expect(html).toContain("Dashboard");
    expect(html).toContain('href="/settings"');
    expect(html).toContain("aria-label");
  });
});

describe("component count", () => {
  it("has 34 components registered", () => {
    const result = handleGenerateMarkup({ component: "nonexistent" });
    const available = result.content[0].text;
    const commaCount = (available.match(/,/g) || []).length;
    expect(commaCount).toBeGreaterThanOrEqual(33);
  });
});
