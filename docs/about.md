# About this project

## Why I built it

*Bennett Landman, MASI Lab, Vanderbilt University*

Once upon a time, I did a lot of programming. Throughout grad school I had at least two terminals open
and computers running at all times. When I joined Vanderbilt, I coded a ton of server logic along with
statistics and model fitting. As time has gone on, my role has become much more mentorship, writing,
editing and service, and much less hands-on coding. Modern development systems also take a lot of work
to keep patched and synchronized, so development environments that go unused tend to rust and their
libraries fall out of date. My GitHub record for the beginning of this year was extremely sparse.

For the last few months I have been using AI coding much more deeply than scripting or debugging other
people's code, and recently I've gained enough confidence in what my code is doing to send it to
GitHub. This MICCAI was the first conference where I could see something cool and quickly check the
possibilities, either with code provided by the authors (thanks!) or with a bit of vibe coding.
`pymodelvis` is one of the tools that came out of agents running while I paid attention to the next talk.

The itch it scratches is an old one. Graph viewers draw a network's *operations*. When I explain a
model to a student, a clinician or a reviewer, I want to show what a specific scan or photograph
*becomes* inside the network: which stages matter, what each one responds to, where the evidence for
the answer lies, and how all of that changes when the input changes. That matters most for the 3-D
medical-imaging networks my lab builds, such as UNesT, where the interesting representations are
volumes, not pictures.

## How it was made

This project was written almost entirely by an AI coding agent, working from my specifications and
feedback:

* **Agent:** Claude, Anthropic's AI model, in the Claude desktop app (Cowork mode)
* **Model:** `claude-opus-5-5` (Claude Opus 5.5), as configured for the sessions that built it
* **Period:** September 2026
* **My role:** the specification, the design direction (what should be shown, and how it should look),
  reviewing every figure, and deciding what to publish
* **The agent's role:** the code, tests, examples, demo-model training, documentation, slide deck and
  this website

The first request was a long specification for a "representation-flow" visualizer. Later rounds asked
for more beauty (the black cinematic style), movies over changing inputs, real pretrained models, a
command-line tool, GitHub-ready documentation and first-class support for 3-D transformer networks.
The [changelog](../CHANGELOG.md) lists what came out of each round.

Some things I did to keep AI-written code trustworthy, and that I recommend if you do the same:

* **Tests for every feature**, run on every push (CPU-only, about two minutes).
* **Look at the output.** Every figure in the gallery was inspected. Several real bugs (a lesion head
  stuck at chance, a Grad-CAM that was zero on the last ViT block, a transformer backbone that was
  silently left out of UNETR figures) were found by looking, not by tests.
* **Be honest about limits.** When something could not be verified, such as running the real UNesT
  weights in the build environment, the documentation says so.

## Credits

* The models shown are the work of their authors; see [References](references.md) for the papers
  behind every demo model and method.
* MONAI, PyTorch, timm, TorchXRayVision, nibabel and nilearn make the examples possible.
* The Inter typeface is by Rasmus Andersson (SIL Open Font License).

## Contact

Issues and pull requests are welcome at
[github.com/BennettLandman/pymodelvis](https://github.com/BennettLandman/pymodelvis). The MASI Lab is at
[my.vanderbilt.edu/masi](https://my.vanderbilt.edu/masi/).
