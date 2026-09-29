# Topology: hooks, runtime dataflow and torch.fx

## Why hooks alone are not enough
Forward hooks give module inputs/outputs and execution order. They cannot see *functional*
operations in a parent's `forward`: `torch.cat([up, skip])`, `x + residual`, `x[:, 0]`. Those are
exactly the operations that create skip connections, fusion and branching.

## Runtime dataflow tracing (default, `topology="auto"`)
During the metadata pass a `torch.overrides.TorchFunctionMode` (`capture.DataflowTracer`) sees
every tensor operation. Each produced tensor is tagged with an op-node id (a Python attribute on
the tensor object). Each op records its parent op ids and the module call active on the hook
stack at the time. Model inputs are passed as detached *views* tagged as input nodes, so user
tensors are never modified. Nothing is retained except small integer records; the tags die with
the tensors.

Stage edges come from walking backwards from each selected stage's input tensors through the op
graph. The walk stops at the first op owned by *another* selected stage (or a model input).
Ownership uses the module call tree: an op belongs to the nearest selected ancestor call.
Model-output edges are found the same way. This recovers:

* skip / residual connections (`enc1 → dec1`), classified as **skip** when the source is also an
  ancestor of the primary path;
* multimodal **merge** edges (`clinical_mlp → fusion`);
* branching heads (one stage feeding several heads);
* reused modules (each call is its own stage).

It also works with data-dependent control flow, because it observes the actual execution. The
*primary* incoming edge of a stage is the deepest pathway (most ancestors), which keeps the trunk
straight in the layout.

If tracing raises (for example, exotic tensor subclasses), the pass is re-run without the tracer,
with a warning, and topology falls back to execution order.

## torch.fx investigation (`topology="fx"`)
`torch.fx.symbolic_trace` can supply topology while hooks supply runtime tensors. With a custom
`Tracer` that treats the selected stage modules as leaves, FX yields a static graph whose
`call_module` nodes are the stages. Propagating "producing stage" sets through the other nodes
gives stage edges. On traceable models (e.g. the test U-Net) this reproduces the runtime edges
exactly.

Findings:
* FX fails on data-dependent Python control flow (`if x.mean() > 0:`), many HuggingFace models, and
  code that inspects tensor values. That is why it is *not* the default.
* FX gives no activations. It complements hooks and cannot replace them.
* On failure `neural_flow` warns *"model could not be symbolically traced … Using runtime execution
  order instead."* and uses the sequential fallback. Activation visualisation is never blocked.

## Sequential fallback (`topology="sequential"`)
Stages are chained in execution order from the first input. Any stage that ended up without a
predecessor in the other modes is also attached to the previous stage, so the figure is always
connected.

## Capture pass
The second pass registers hooks only on modules that own a selected stage call, counting calls
per module to hit the right invocation of reused modules. Summaries are computed inside the hook
(see `tensor_rendering.md`), so large activations are reduced before anything moves to
`capture_device`. An `AttentionTracer` mode records attention weights from
`F.multi_head_attention_forward` / `F.scaled_dot_product_attention` and attributes them to the
enclosing selected stage. All hooks and modes are removed in `finally` blocks, and the model's
`training` flag is restored.
