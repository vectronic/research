### TensorFlow

An Estimator is TensorFlow's high level representation of a complete model. 
TensorFlow provides a collection of pre-made Estimators 

You must create input functions to supply data. 
We recommend using TensorFlow's Dataset API, which can deftly parse all sorts of data.

Checkpoints to save and restore models.
Estimators automatically write the following to disk:
- checkpoints, which are versions of the model created during training.
- event files, which contain information that TensorBoard uses to create visualizations.

A trainable model must modify the values in the graph to get new outputs with the same input.
Layers are the preferred way to add trainable parameters to a graph.

We divide the task of supporting a file format into two pieces:
- File formats: We use a Reader Op to read a record (which can be any string) from a file.
- Record formats: We use decoder or parsing Ops to turn a string record into tensors usable by TensorFlow.

The main steps for a new Op are:
- Registering the op.
- Define and register an OpKernel.
- Add the Python wrapper. 

There are several reasons why you might want to create a custom C++ op:
- It's not easy or possible to express your operation as a composition of existing ops.
- It's not efficient to express your operation as a composition of existing primitives.
- You want to hand-fuse a composition of primitives that a future compiler would find difficult fusing.

The XLA Compiler has an experimental implementation of automatic kernel fusion.

A GPU kernel is implemented in two parts: the OpKernel and the CUDA kernel and its launch code.
- an op registration can have multiple inputs and outputs.

Client
- Defines the computation as a dataflow graph.
- Initiates graph execution using a session

Distributed Master
- Prunes a specific subgraph from the graph, as defined by the arguments to Session.run().
- Partitions the subgraph into multiple pieces that run in different processes and devices.
- Distributes the graph pieces to worker services.
- Initiates graph piece execution by worker services.

Worker Services (one for each task)
- Schedule the execution of graph operations using kernel implementations appropriate to the available hardware (CPUs, GPUs, etc).
- Send and receive operation results to and from other worker services.

Kernel Implementations
- Perform the computation for individual graph operations.

Note that the Distributed Master and Worker Service only exist in distributed TensorFlow. The single-process version of TensorFlow includes a special Session implementation that does everything the distributed master does but only communicates with devices in the local process.

Master:
Since the master sees the overall computation for a step, it applies standard optimizations such as common subexpression elimination and constant folding.
- prunes the graph to obtain the subgraph required to evaluate the nodes requested by the client,
- partitions the graph to obtain graph pieces for each participating device
	- The distributed master has grouped the model parameters in order to place them together on the parameter server
	- Where graph edges are cut by the partition, the distributed master inserts send and receive nodes to pass information between the distributed tasks 
	- The distributed master then ships the graph pieces to the distributed tasks.
- caches these pieces so that they may be re-used in subsequent steps.

The worker service in each task:
- handles requests from the master,
- schedules the execution of the kernels for the operations that comprise a local subgraph, and
- mediates direct communication between tasks.
- The worker service dispatches kernels to local devices and runs kernels in parallel when possible, for example by using multiple CPU cores or GPU streams
- We specialize Send and Recv operations for each pair of source and destination device types:
	- Transfers between local CPU and GPU devices use the cudaMemcpyAsync() API to overlap computation and data transfer.
	- Transfers between two local GPUs use peer-to-peer DMA, to avoid an expensive copy via the host CPU.
	- For transfers between tasks, TensorFlow uses multiple protocols, including:
		- gRPC over TCP.
		- RDMA over Converged Ethernet.


Op docs: https://www.tensorflow.org/api_docs/python/tf/equal

Architecture: https://www.tensorflow.org/extend/architecture

Checkpoints: https://www.tensorflow.org/get_started/checkpoints

### RaftLib

The Raft Library is built to bring together systems programming techniques and algorithms from the high performance computing community and apply them to the big data analytics world that is currently dominated by very high level language constructs. 

The Raft parallel runtime system begins by dividing compute into stateful classes called "kernels." Each kernel should contain all the state that is needed for computation.

The Raft library handles the allocates for the programmer based on a port interface. When the programmer specifies a kernel connection:

The runtime decides first where to run the kernels a & b, then decides what type of memory to allocate. 

Nice expression for parallel processing;

```
    /**
     * detect # output ports from reader,
     * duplicate c that #, assign each
     * output port to the input ports in
     * writer
     */
    m += reader <= c >= writer;
```
Concepts:
- allocate types: shmem, heap. Allocate a FIFO between src and dest ports, dynamic monitor and resize fifos based in stats or ignore if fixed alloc
- core affinity, partition
- basicparallel: adds parallel cloned kernel if stats > 0.5
- scheduler
- ringbuffer
    - commands?
    - tuples?
- data and datamanagers
- launch allocator in thread, launch scheduler in thread, start parallelism_monitor
- kernel, kernel container and commands in queues
- data types are fixed on ports, allows allocation of known sized elements

### Media-IO

Media-IO is a framework to process media, including video, audio, subtitle, etc.

Its architecture is plugin oriented to offer the ability to easily connect new features in various languages, such as C++, Rust, and others.

- Reader: interface to read data like FileSystem or over HTTP
- Unwrapper: provide access to packets of data for each stream
- Decoder: give the capability to decode a coded stream to raw images
- Generator: generate images based on chart definitions, equations, etc.
- Filter: take image to return a processed version
- Analyser: return computed information based on image(s)
- Encoder: generate the coded stream
- Wrapper: mux coded streams
- Writer: store wrapped data on storage

An important fact is the facility to create new plugin. Each process can be defined in different language (C, C++, Rust, D, etc.) as each one can provide the C ABI required for communication.

Concepts:
- metadata: types, names, hierarchies
- file: number of streams, stream types
- plugin: repository cache, loading, identifier, interfaces
- codecs
- framework: interface types and instances: reader, writer, generator, wrapper, unwrapper, encoder, decoder, filter, analyser
- data: image->component data, audio->channel data, coded data
- reader: orchestrates instancing, configuration and calls to plugin instances
- player: uses reader and queue/timing/threads to process reading file and displaying it

### Disruptor

- Disruptor is an efficient multi-queue for very high throughput messaging
- With concurrency, the difficulty comes because these threads have to communicate with each other. Processing an order changes market conditions and these conditions need to be communicated. processors spent more time managing queues than doing the real logic of the application. Queue access was a bottleneck.
- To get the best caching behavior, you need a design that has only one core writing to any memory location. Multiple readers are fine, processors often use special high-speed links between their caches. But queues fail the one-writer principle.

### Microsoft Media Foundation

- Sessions and Pipelines
- Why do MS differentiate between in the loop and a transform in pipeline => This programming model gives the application more control over the flow of data, and also gives the application direct access to the data from the source. The pipeline is end-to-end, meaning it handles data flow from the source (such as a video file) all the way to the destination (such as the graphics display). However, if you want to read or modify the data as it goes through the pipeline, you must write a custom plug-in. That requires a fairly deep knowledge of the Media Foundation pipeline. For certain tasks, creating a new plug-in is too much overhead. The source reader is designed for this type of situation, when you want to get the raw data from a source without the overhead of the entire pipeline.
- use source/sink/transform interfaces directly to be ‘in the loop’ or use a session to run the pipeline

### Parallelization for Fork/Join

1. To start with, you need a substantial chunk of CPU-intensive processing, all happening on a single collection of data. Parallelization implies overheads (task coordination, handover from thread to thread, etc.) and won't help you to speed up all those little bits of processing here and there. Needless to say, tasks where anything but the CPU is the bottleneck will not benefit from parallelization.
2. It needs to be stateless: the processing of an element should not share any mutable state with the processing of any other element. Also, the processing of one item should not depend on the results of processing any other. Good parallelism asks for thread isolation, with the least possible amount of coordination.
3. The data should support low-latency random access. This is what the Fork/Join framework likes most: it wants to split your collection down the middle, then repeat that for each half, as many (and as few) times needed to get the least possible number of chunks required to saturate all CPU cores. This constraint leaves little choice besides storing everything in an on-heap array.

### Upipe

Upipe was specified bottom-up. It implies that an application using Upipe is not restricted to a single high-level API, but can also implement and use any of the lower-level programming interfaces. It is also always possible to build higher-level interfaces. Its core data structures are expected to work efficiently without needlessly depending on each other,

None of the core modules of Upipe deal with threads. However, the basic data structures have been implemented with lock-less or wait-less algorithms and therefore do not depend on the number of threads. Upipe's philosophy is to let the application (or higher-level APIs) create FIFOs and threads when it is necessary, not when the developer of the framework found it fancy to put a thread there

Since it doesn't require its own thread architecture, it also doesn't require a specific event loop manager.

Another key design choice is the separation of data processing from decision taking. "Pipes" are not supposed to make any decision (such as selecting this elementary stream or decoding to 720x576), and whenever an action is necessary, an event is sent to "probes" provided by the application, which would typically send a control command to the pipe, or dynamically create (recreate) a part of the pipeline by allocating new pipes.
At the core of the ability to re-allocate buffers between compute kernels, is an efficient lock free mechanism to do so. The current solution is initiated through the dynamic allocator thread, however it is mediated through several intervening layers to maintain type agnosticism. Basically the top layer of RaftLib (the user facing layer) knows about types. The underlying run-time itself only cares about bytes. This makes developing on one end or the other rather easy. This also means that re-allocating is a bit tricky. Mediating the re-allocation and orchestrating the handoff between producer, consumer, and allocator is they data manager class which is the wrapper for each buffer object within every port.

Pipes and applications however do not manipulate struct ubuf directly, but struct uref, which is composed of a pointer to struct ubuf and a pointer to udict. The dictionary allows to associate arbitrary attributes to the struct ubuf.

When created, pipes are passed one structure for logging and sending exceptions.

Data is fed into a pipe using upipe_input.

Control commands and pipes are classified (that is, enum values are prepended with a prefix) into 7 categories:
- generic: commands and probes which can apply to any type of pipe (no prefix)
- source: for pipes that have no input, but rely instead on external events to retrieve incoming data
- join: for pipes that have several inputs, such as a mux
- split: for pipes that have several outputs, such as a demux
- sink: for pipes that have no output and may rely on external events
- void: for pipes that have neither input nor output (such a pipe may be used internally to create other pipes)
- pipe type-specific commands and probes which must be prefixed with the short name of the pipe

External events: Source pipes and sink pipes (but not exclusively) rely on external events to retrieve or dispatch data. For instance, one may want to wait on a UDP socket for packets. Or to wait until a system pipe (mkfifo) can be written again. Or more simply, wait for a timeout.

Internal events: Buffer management structures are allocated on the fly, depending on the needs of the pipes. However, some negotiation may take place, because downstream pipes may have specific requirements, such as the alignment of data, or even a custom ubuf manager required by an external library. 

Flows: The movement of buffers between the input and output of a pipe is called a flow. Pipes generally expect some parameters describing the flow; these are called a flow definition. The input flow definition is set on a pipe using upipe_set_flow_def. It may be changed at any time, but the pipe may deny it by returning an error if it considers it is too big a change; by convention, pipes only allow it if it doesn't require a full reset of the pipe's state. Some pipes (filters) may also require a flow definition describing the requested output, for instance to change picture format.

It is useful to note that a flow definition packet is actually a patchwork of attributes with different purposes:
- attributes which define the format of the buffers, such as the number of planes or chroma subsampling
- attributes which describe several properties useful for the display or processing of the flow, such as aspect ratio, fps, bitrate, sample rate, etc.
- attributes which may have no relation with the data themselves but are useful to choose between elementary streams, such as the language, program name, event information, etc.

Threads: Upipe objects do not natively deal with threads. Multithreading is supposed to be the prerogative of the application, or at least of very high level bin pipes (an exception being the libraries which themselves take advantage of multithreading, such as FFmpeg/libav and x264). 

### OpenVX

- Everything in a `context`
- Processing of `data` within a `graph` of `nodes`, each of which is a `kernel` that runs on a specified `target`

### GStreamer

A pad can have any of three availabilities: always, sometimes and on request. The meaning of those three types is exactly as it says: always pads always exist, sometimes pad exist only in certain cases (and can disappear randomly), and on-request pads appear only if explicitly requested by applications.

### Avid

The Avid Intelligent Architecture will evaluate the platform configuration, OS, hardware, and GPU capabilities in the system. Based on the availability of processors for the various compute hardware, it will execute them in an optimal pipelined and parallel processing manner. It will dynamically distribute the processing to the device best suited to the specific task for different segments of the timeline.

### Flow Based Programming

- In terms of encoding, a dataflow program might be implemented as a hash table, with uniquely identified inputs as the keys, used to look up pointers to the instructions. When any operation completes, the program scans down the list of operations until it finds the first operation where all inputs are currently valid, and runs it. When that operation finishes, it will typically output data, thereby making another operation become valid.
- For parallel operation, only the list needs to be shared; it is the state of the entire program. Thus the task of maintaining state is removed from the programmer and given to the language's runtime.

### Reactive and Back-Pressure

- Handling streams of data—especially “live” data whose volume is not predetermined—requires special care in an asynchronous system. The most prominent issue is that resource consumption needs to be controlled such that a fast data source does not overwhelm the stream destination. Asynchrony is needed in order to enable the parallel use of computing resources, on collaborating network hosts or multiple CPU cores within a single machine.

- The main goal of Reactive Streams is to govern the exchange of stream data across an asynchronous boundary: think passing elements on to another thread or thread-pool while ensuring that the receiving side is not forced to buffer arbitrary amounts of data. In other words, back pressure is an integral part of this model in order to allow the queues which mediate between threads to be bounded. The benefits of asynchronous processing would be negated if the communication of back pressure were synchronous 

### Stream Processing

- Stream processing is especially suitable for applications that exhibit three application characteristics:

	- Compute Intensity, the number of arithmetic operations per I/O or global memory reference. In many signal processing applications today it is well over 50:1 and increasing with algorithmic complexity.
	- Data Parallelism exists in a kernel if the same function is applied to all records of an input stream and a number of records can be processed simultaneously without waiting for results from previous records.
	- Data Locality is a specific type of temporal locality common in signal and media processing applications where data is produced once, read once or twice later in the application, and never read again. Intermediate streams passed between kernels as well as intermediate data within kernel functions can capture this locality directly using the stream processing programming model.

Optimization concepts:

- micro-batching and latency reduction
- load balancing across data partitions
- thread scheduling on cores and communication
- thread sync via FIFOs
- thread pool
- partitioning and aggregation costs
- predictive load balancing via parallelization based on operation cost
- parallelization overhead such as slower clock speed when using multiple cores
- remote processing
- kernel aggregation
- remote IPC aggregation
- tilable kernels

### Spring Integration

Spring Integration is motivated by the following goals:

- Provide a simple model for implementing complex enterprise integration solutions.
- Facilitate asynchronous, message-driven behavior within a Spring-based application.
- Promote intuitive, incremental adoption for existing Spring users.

Spring Integration is guided by the following principles:

- Components should be loosely coupled for modularity and testability.
- The framework should enforce separation of concerns between business logic and integration logic.
- Extension points should be abstract in nature but within well-defined boundaries to promote reuse and portability.

In Spring Integration, a Message is a generic wrapper for any Java object combined with metadata used by the framework while handling that object. It consists of a payload and headers. The payload can be of any type and the headers hold commonly required information such as id, timestamp, correlation id, and return address. Headers are also used for passing values to and from connected transports. For example, when creating a Message from a received File, the file name may be stored in a header to be accessed by downstream components. 

A Message Channel represents the "pipe" of a pipes-and-filters architecture. Producers send Messages to a channel, and consumers receive Messages from a channel. The Message Channel therefore decouples the messaging components, and also provides a convenient point for interception and monitoring of Messages.

A Message Endpoint represents the "filter" of a pipes-and-filters architecture. As mentioned above, the endpoint’s primary role is to connect application code to the messaging framework and to do so in a non-invasive manner. In other words, the application code should ideally have no awareness of the Message objects or the Message Channels. 

A Channel Adapter is an endpoint that connects a Message Channel to some other system or transport. 

While the Message plays the crucial role of encapsulating data, it is the MessageChannel that decouples message producers from message consumers.

Channel Interceptors: One of the advantages of a messaging architecture is the ability to provide common behavior and capture meaningful information about the messages passing through the system in a non-invasive way. Since the Message s are being sent to and received from MessageChannels, those channels provide an opportunity for intercepting the send and receive operations.

While the Message plays the crucial role of encapsulating data, it is the MessageChannel that decouples message producers from message consumers.

### DSL NOTES

- Some example DSLs:
    - http://highlandjs.org
    - http://zulko.github.io/moviepy/getting_started/quick_presentation.html
- stream creation:
	- stream constructor passed an array, generator function, native stream/queue etc to wrap, iterator or event source
	- stream.pause()/resume()/end()
- stream transforms:
	- .append(y)
	- .batch(n)
	- .batchWithTimeOrCount(ms, n)
	- .consume(f) - used as basis for .filter(f)
	- .doto(f) alias of .tap(f) -> compare to .map(f)
	- .invoke(method, args)
	- .find(f) - first value of a .filter() 
	- .where(props) 
	- .findWhere(props) - first value of a .where() 
	- .group(f)
	- .ratelimit(num, ms)
	- .reduce(memo, iterator)
	- .split()
- higher order streams operations:
	- .concat(ys)
	- .flatFilter(f)
	- .flatMap(f)
	- .flatten() 
	- .fork()
	- .merge()
	- .otherwise(ys)
	- .zip(ys)
- stream consumption:
	- .each(f)
	- .done(f)
	- .pipe(dest, options) to a non-stream based output

Example code:
```
const source = getAsyncStockData();
const subscription = source
  .filter(quote => quote.price > 30)
  .map(quote => quote.price)
  .forEach(price => console.log(`Prices higher than $30: ${price}`));
```