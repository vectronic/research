### Core Framework Criteria

- scriptable processing logic, routing and configuration
- frontend scripting abstracted from backend processing framework
- templatable scripts for runtime customisation
- implement operators in script or use externally implemented plugin operators
- dynamic operator loading
- dynamic online control of processing framework parameters
- browser integration for UI
- support heterogeneous resources
- support for live/real-time processing
- hierarchical composition of processing graphs
- optimise throughput and latency for single stream with a single job on a single heterogeneous system
- optimise for utilisation, throughput and latency for multiple different rate streams with different complexity jobs on a single heterogeneous system
- intra-process and inter-process kernels
- well documented and tested
- simple install and usage

### Nextflow

\+ templatable DSL

\+ flexible

\- needs Java

\- all processing via processes

\- no plugins/dynamic libs

\- no browser integration

\- no media concepts e.g. live, frames

### ScalaPipe

\+ DSL

\+ heterogeneous resource support

\- no browser integration

\- needs Scala

\- no plugins/dynamic libs

\- no media concepts e.g. live, frames

### Media-IO

\+ plugins/dynamic libs

\+ polyglot

\+ simple

\- immature

\- no browser integration

\- no scripting or DSL

\- no heterogeneous resource support

\- API is too restrictive

\- no real pipeline runtime

### Upipe

\+ media based concepts e.g. live, frames

\- minimal documentation

\- not polyglot

\- no dynamic plugins

\- no browser integration

\- no scripting or DSL

\- no heterogeneous resource support

\- want more dynamic scheduling and control of execution on threads etc.

### GStreamer

\+ live/faster than real-time

\+ plugins

\+ media based concepts

\+ polyglot

\+ mature

\- too much locking

\- complex

\- hard to abstract away from

\- no back-pressure 

\- no browser integration

\- no scripting 

\- no dynamic plugins

### MLT

\+ plugins (modules) for ffmpeg etc.

\+ includes timeline model

\- no browser integration

\- no scripting 

\- too rigid in frame structure

\- lacking first class multi-scalar support

### RaftLib

\+ lots of effort into the core stream processing concepts e.g. online optimisation, focused on multi-scalar support
 
\+ online processing optimisation

\- lacking heterogeneous resource support (although intended)

\- immature

\- no browser integration

\- no scripting, DSL

\- no plugins/dynamic libs

\- no media concepts e.g. live, frames

\- fixed types for stream elements

### PipeFabric

\+ accessible, neat, documented

\+ nice table<->stream concept

\+ lots of effort into the core stream processing concepts e.g. focused on multi-scalar support

\+ nice DSL

\- no heterogeneous hardware concepts

\- no browser integration

\- no plugins/dynamic libs

\- no media concepts e.g. live, frames

\- fixed tuples for stream elements
