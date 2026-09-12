# Cellaris marker overlay for EOV

A plugin for EOV based on the Cellaris model, a histology project that asks which of the 19 Orion
multiplex-IF channels are predictable from an H&E whole-slide image.

Marker heads have been trained on 15 and on 41 patients. What they learn beyond a tissue-density
control is small, uneven across markers, and bounded by something we can now measure.

Every number is scored against a tissue-density control: a predictor that knows nothing but optical
density. It reaches 0.71 AUC on an unstained channel, so a raw AUC near 0.75 can mean nothing was
learned. Read the gain over the control, never the AUC itself.

## The plugin

It reads level-0 tiles under the viewport through the host, predicts them on a worker thread, and
paints three markers over the H&E: Hoechst in blue, Pan-CK in green, SMA in amber. Those are the
three that clear the control on the Orion cohort. CD8a was in the first version and came out again,
because it does not.

![The overlay in EOV, filling in window by window over a colorectal slide](image.png)

## What it inherits from the measurements

The model was trained on 256px inputs and does not generalise past them: a 384px input puts a ring
of +2 arcsinh around every tile. So inference runs at 256 with a 128px stride and feathered
blending, which measures 0.91 at the joins against 4.88 unblended and 22.7 for a 384px window. Past
600 tiles in view it declines and frames the view in amber rather than predicting at a scale the
model never saw.

One window takes about 1.2 seconds on a CPU, so a view at 1:1 fills in over a few minutes. There is
no GPU path yet.

## What it actually shows

On its own the overlay looks convincing. Next to the measured IF on a patient the model never saw,
it is a nuclear stain with a tint.

![H&E, prediction and measured Orion IF for four CRC11 tiles](image-1.png)

Pixel correlation between predicted and measured on those four tiles runs +0.19 to +0.91 for
Hoechst, -0.07 to +0.21 for Pan-CK and -0.10 to +0.32 for SMA. Hoechst is the part haematoxylin
gives away, since haematoxylin and Hoechst both bind nucleic acid. Pan-CK comes out as a wash where the
measurement is patchy, and SMA is missing from stroma that measures full of it.

The model, and how good it is, live in cellaris.

## Running it

Install EOV. Put cellaris.eop in ~/.eov/plugins/, and markers.onnx and onnxruntime.dll in
~/.eov/plugins/cellaris/ beside it. Open a slide, click the button in the toolbar, and zoom in until
the amber frame goes away.

CELLARIS_MODEL and CELLARIS_ORT_DYLIB override those two paths if you keep your models elsewhere. A
path that does not exist is reported rather than skipped.

Any export from cellaris works. Markers are resolved by name against the channel order recorded in
the model file, and a model that carries no channel order is refused rather than guessed at. An
earlier version hard-coded the indices and painted Ki67 as Pan-CK.

## Building

One clone, and nothing else checked out. The API comes from eov-plugin-api on crates.io.

    cargo build --release
    python package.py --onnxruntime path/to/onnxruntime.dll

Built against eov-plugin-api 0.4, and verified loading in the EOV v0.3.1 release binary.

## References

The data the model was trained on, and the two conventions the overlay borrows. The rest — the
controls, the ceiling, the patient bootstrap, the cross-cohort work — are in cellaris.

Lin J-R, Chen Y-A, Campton D, Cooper J, Coy S, Yapp C, Tefft JB, McCarty E, Ligon KL, Rodig SJ,
Reese S, George T, Santagata S, Sorger PK. High-plex immunofluorescence imaging and traditional
histology of the same tissue section for discovering image-based biomarkers. Nature Cancer (2023).
doi:10.1038/s43018-023-00576-1 — the Orion CRC cohort.

Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation.
MICCAI (2015). arXiv:1505.04597 — the marker head.

Kingma DP, Welling M. Auto-encoding variational Bayes. ICLR (2014). arXiv:1312.6114 — the shared
encoder.

Bendall SC, Simonds EF, et al. Single-cell mass cytometry of differential immune and drug responses
across a human hematopoietic continuum. Science 332, 687—696 (2011). doi:10.1126/science.1198704 —
the arcsinh transform with cofactor 5, used to display every intensity here.

Built on EOV, which carries its own licence and history.

## License

Apache-2.0, MIT or GPLv3, the same three EOV offers, at your option. The plugin package bundles ONNX
Runtime, which is MIT.
