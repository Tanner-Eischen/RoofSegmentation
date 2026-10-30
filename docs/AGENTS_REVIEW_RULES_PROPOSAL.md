# Proposed AGENTS.md additions from label-review verification

These additions are proposed, not applied. The Codex-md-update skill specifies
"Update AGENTS.md upon confirmation." The repository's existing global
instructions remain unchanged.

- Verify every correction polygon on a single original image at native
  resolution. Remove montage/header offsets from coordinates; never infer
  extra image extent across repeated side-by-side panels.
- Before writing masks, reject aliases to preserved inputs or other writable
  files, including hardlinks to source files outside the review folder. Validate
  the complete edit batch before changing any files.
- Preserve recorded human adjudication. AI proposals remain explicitly
  provisional and cannot overwrite a human-approved draft. Confirmed-label
  reports require supplied reviewer decisions and class-policy provenance.
