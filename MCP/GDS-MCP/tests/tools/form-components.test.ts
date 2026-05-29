import { describe, it, expect } from "vitest";
import { handleGenerateMarkup } from "../../src/tools/generate-markup.js";
import { handleSuggestComponent } from "../../src/tools/suggest-component.js";
import { handleReviewHtml } from "../../src/tools/review-html.js";

describe("textarea", () => {
  it("generates textarea markup", () => {
    const result = handleGenerateMarkup({
      component: "textarea",
      options: { label: "Can you provide more detail?", id: "more-detail" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-textarea");
    expect(html).toContain('for="more-detail"');
    expect(html).toContain('rows="5"');
  });

  it("is suggested for multi-line text needs", () => {
    const result = handleSuggestComponent({ useCase: "enter longer text feedback" });
    expect(result.content[0].text).toContain("Textarea");
  });
});

describe("checkboxes", () => {
  it("generates checkboxes with fieldset and legend", () => {
    const result = handleGenerateMarkup({
      component: "checkboxes",
      options: {
        label: "Which types of waste do you transport?",
        id: "waste",
        items: [
          { text: "Waste from animal carcasses", value: "carcasses" },
          { text: "Waste from mines or quarries", value: "mines" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-fieldset");
    expect(html).toContain("<legend");
    expect(html).toContain("Which types of waste");
    expect(html).toContain("govuk-checkboxes");
    expect(html).toContain('value="carcasses"');
    expect(html).toContain('for="waste-1"');
    expect(html).toContain('id="waste-1"');
  });

  it("is suggested for multi-select needs", () => {
    const result = handleSuggestComponent({ useCase: "select multiple options from a list" });
    expect(result.content[0].text).toContain("Checkboxes");
  });

  it("review flags checkboxes without fieldset", () => {
    const html = `<div class="govuk-checkboxes">
  <input type="checkbox" id="a" value="a"><label for="a">A</label>
  <input type="checkbox" id="b" value="b"><label for="b">B</label>
</div>`;
    const result = handleReviewHtml({ html, component: "checkboxes" });
    expect(result.content[0].text).toContain("fieldset");
  });
});

describe("radios", () => {
  it("generates radios with fieldset", () => {
    const result = handleGenerateMarkup({
      component: "radios",
      options: {
        label: "Where do you live?",
        id: "where",
        name: "where-do-you-live",
        items: [
          { text: "England", value: "england" },
          { text: "Scotland", value: "scotland" },
          { text: "Wales", value: "wales" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-radios");
    expect(html).toContain("govuk-fieldset");
    expect(html).toContain("Where do you live?");
    expect(html).toContain('value="england"');
    expect(html).toContain('name="where-do-you-live"');
  });

  it("generates inline radios", () => {
    const result = handleGenerateMarkup({
      component: "radios",
      options: {
        label: "Have you changed your name?",
        variant: "inline",
        items: [
          { text: "Yes", value: "yes" },
          { text: "No", value: "no" },
        ],
      },
    });
    expect(result.content[0].text).toContain("govuk-radios--inline");
  });

  it("review warns on pre-selected radio", () => {
    const html = `<fieldset class="govuk-fieldset">
  <legend class="govuk-fieldset__legend">Choice</legend>
  <div class="govuk-radios">
    <input type="radio" id="r1" name="choice" value="a" checked>
    <label for="r1">A</label>
    <input type="radio" id="r2" name="choice" value="b">
    <label for="r2">B</label>
  </div>
</fieldset>`;
    const result = handleReviewHtml({ html, component: "radios" });
    expect(result.content[0].text).toContain("pre-selected");
  });
});

describe("character-count", () => {
  it("generates character count markup", () => {
    const result = handleGenerateMarkup({
      component: "character-count",
      options: { label: "Provide details", id: "details", maxlength: "500" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-character-count");
    expect(html).toContain('data-maxlength="500"');
    expect(html).toContain("govuk-js-character-count");
    expect(html).toContain("details-info");
    expect(html).toContain("You can enter up to 500 characters");
  });
});

describe("date-input", () => {
  it("generates date input with day/month/year fields", () => {
    const result = handleGenerateMarkup({
      component: "date-input",
      options: {
        label: "When was your passport issued?",
        hint: "For example, 27 3 2007",
        id: "passport-issued",
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-date-input");
    expect(html).toContain('role="group"');
    expect(html).toContain("passport-issued-day");
    expect(html).toContain("passport-issued-month");
    expect(html).toContain("passport-issued-year");
    expect(html).toContain('inputmode="numeric"');
    expect(html).toContain("govuk-input--width-2");
    expect(html).toContain("govuk-input--width-4");
  });
});

describe("file-upload", () => {
  it("generates file upload markup", () => {
    const result = handleGenerateMarkup({
      component: "file-upload",
      options: { label: "Upload a photo", id: "photo" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-file-upload");
    expect(html).toContain('type="file"');
    expect(html).toContain('for="photo"');
  });
});

describe("password-input", () => {
  it("generates password input with show/hide toggle", () => {
    const result = handleGenerateMarkup({
      component: "password-input",
      options: { label: "Create a password", id: "password" },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-password-input");
    expect(html).toContain('type="password"');
    expect(html).toContain('spellcheck="false"');
    expect(html).toContain("Show password");
    expect(html).toContain('aria-controls="password"');
  });
});

describe("select", () => {
  it("generates select dropdown", () => {
    const result = handleGenerateMarkup({
      component: "select",
      options: {
        label: "Sort by",
        id: "sort",
        items: [
          { text: "Recently published", value: "published" },
          { text: "Recently updated", value: "updated" },
        ],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-select");
    expect(html).toContain("<option");
    expect(html).toContain('value="published"');
    expect(html).toContain("Recently published");
  });
});

describe("error states across components", () => {
  it("textarea with error includes visually hidden prefix", () => {
    const result = handleGenerateMarkup({
      component: "textarea",
      options: {
        label: "Details",
        errorMessage: "Enter more detail",
        id: "details",
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-form-group--error");
    expect(html).toContain("govuk-visually-hidden");
    expect(html).toContain("Error:");
    expect(html).toContain("govuk-textarea--error");
    expect(html).toContain('aria-describedby="details-error"');
  });

  it("checkboxes with error includes error in aria-describedby", () => {
    const result = handleGenerateMarkup({
      component: "checkboxes",
      options: {
        label: "Select at least one",
        errorMessage: "Select an option",
        id: "opts",
        items: [{ text: "Option A", value: "a" }],
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-form-group--error");
    expect(html).toContain('aria-describedby="opts-error"');
    expect(html).toContain("govuk-visually-hidden");
  });

  it("file upload with error", () => {
    const result = handleGenerateMarkup({
      component: "file-upload",
      options: {
        label: "Upload evidence",
        errorMessage: "Select a file",
        id: "evidence",
      },
    });
    const html = result.content[0].text;
    expect(html).toContain("govuk-file-upload--error");
    expect(html).toContain("govuk-visually-hidden");
  });
});
