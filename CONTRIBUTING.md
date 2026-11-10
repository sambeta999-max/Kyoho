# Contributing to Kyoho

Kyoho is a small fork of [Eden](https://git.eden-emu.dev/eden-emu/eden) that works on Switch 2 support. It has no team, no chat server and takes no donations. If you want to support the people who built almost all of this emulator, support Eden.

## Reporting problems

Open an issue with the bug report template. A report is useful when it has:

- the build (the commit hash or release name in the title bar), the game and its version;
- your CPU, GPU and graphics driver version;
- the settings you changed, and the steps to reach the problem;
- a screenshot, and the log of that session (check it for personal information first).

Do not attach games, keys, firmware or anything else you would not want public.

Problems that also happen in Eden, and Switch 1 games in general, are better reported to Eden: a fix there helps more people, and Kyoho picks it up from upstream.

## Code

Kyoho is licensed under the GPLv3 (or any later version), like the projects it comes from.

- Keep changes focused, and follow the style of the code around them. Eden's [coding guidelines](docs/policies/Coding.md) apply.
- Say what you tested: the game, the scene and the hardware. "It builds" and "the unit tests pass" are worth stating too, but they do not show that a game renders correctly.
- When you port code from another project (Eden, Eglantine, Lichii, Fritchii and others), name the source commit and files in the commit message, check that its license allows it, and keep its copyright notices.
- Disclose AI assistance in the commit message. Most of this fork's own history was written with an AI assistant and says so.
- A fix that is not specific to Switch 2 or to this fork belongs in Eden first.

The documents under [docs/policies](docs/policies) are Eden's and are kept as they came. Where they differ from this file (for example on AI assistance), this file is what applies to Kyoho.

## Building

See the [build guide](docs/Build.md). The inherited documentation still says Eden in most places; the GUI build target is `yuzu` and its output is `kyoho` (`kyoho.exe` on Windows).
