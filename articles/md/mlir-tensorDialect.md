# tensorDialect

### MLIR Tensor 方言详解 

在 MLIR 中，`tensor`​ 方言是进行高级张量操作和形状操作的核心方言。它采用**值语义（Value Semantics）** ，这意味着张量是不可变的（immutable）。任何操作都不会修改原有的张量，而是生成一个新的张量（SSA Value）。

以下是对您提供的代码（典型 Tiling 算子切分场景）中常用 Op 和接口的详细介绍，并额外补充了该方言其他常用操作。

#### 1. 核心 Tiling 操作 Op

这些 Op 是实现张量分块（Tiling）逻辑的基础。

- **​`tensor::EmptyOp`​**

  - **作用**：创建一个没有任何有效数据、仅指定形状和元素类型的“空”张量。
  - **为什么需要它**：在 MLIR 的较新设计中，推荐使用**目标传递风格（Destination-Passing Style, DPS）** 。为了计算一个算子，需要预先定义好其输出张量的形状和结构。`EmptyOp`​ 通常作为这个“初始化/占位”的输出张量（`init`​ 或 `outs`​ 操作数），传递给后续的算子（例如代码中的 `torq_hl::Conv2DOp`）。
  - **特性**：由于它只是形状的占位符，在后续的 Bufferization（张量转内存缓冲区）阶段，它通常会被转化为直接分配内存（`memref.alloc`），或者在可以复用内存时被直接折叠掉。
- **​`tensor::InsertSliceOp`​**

  - **作用**：将一个小源张量（Source Slice）“插入”到一个大目标张量（Destination Tensor）的指定区域，并返回一个被更新过的新大张量。
  - **参数**：

    - `source`：要插入的小张量。
    - `dest`：大目标张量。
    - `offsets`​、`sizes`​、`strides`：定义了对大张量进行插入操作的起始偏移、尺寸和步长。
  - **值语义特性**：在循环中，每次调用 `InsertSliceOp`​ 都会返回一个“新”的 `outputTensor`​。这个新张量包含了刚刚插入的 Tile 小块，并在下一次循环中作为下一个 `InsertSliceOp`​ 的 `dest`​。最终，循环结束时得到的 `outputTensor` 就是拼接完整的最终结果。在编译器后端，这一连串的 Insert 操作会被优化为对同一块连续物理内存的局部写入（In-place update），从而避免真实发生多次全尺寸拷贝。
- **​`tensor::ExtractSliceOp`​**​ （与 `InsertSliceOp` 对偶）

  - **作用**：与 `InsertSliceOp`​ 反向的操作。它从一个大的张量中，根据 `offsets`​、`sizes`​、`strides` 提取出一个小张量（Slice/Tile）。
  - **用法**：在 Tiling 中，通常先用 `ExtractSliceOp`​ 把大 Input 切成小 Input，然后在小 Input 上做计算，最后用 `InsertSliceOp` 把小 Output 塞回大 Output。

#### 2. 通用接口 (Interfaces)

在 MLIR 源码和 Tiling 逻辑中，这些接口为算子提供了统一的特性，方便进行转化和优化。

- **​`DestinationStyleOpInterface`​**​  **(DPS)**

  - 这是前面提到的“目标传递风格”接口。实现了该接口的 Op，其操作数会被明确区分为 `inputs`​ 和 `outputs`。这种明确的分区使得 Bufferization 分析能够更容易地理解 Op 的内存行为，从而进行更有效的内存优化。
- **​`TilingInterface`​**

  - 该接口允许算子基于指定的 `tile sizes`​ 生成自身的切分逻辑。如果一个标准算子实现了此接口，通常可以直接调用 `linalg::tileToForallOp` 等通用工具函数进行切分，而无需像手动编写 C++ For 循环那样处理。
- **​`ViewLikeOpInterface`​**​  **/**  **​`SubsetOpInterface`​**

  - 这些接口标志着一个 Op 是某种视角的提取或插入操作（比如 `ExtractSlice`​ / `InsertSlice`），这有助于优化器分析它是否是对某个大张量的子集操作，从而消除不必要的拷贝（Copy Elision），提升性能。

#### 3. 其他常用 Op

除了 Tiling 场景，`tensor` 方言在降级、形状变换等环节也经常使用到以下 Op：

- **​`tensor::CastOp`​**

  - 用于转换张量的类型约束，通常在静态形状和动态形状之间进行转换。例如，将 `tensor<4x?xf32>`​ 转换为 `tensor<4x8xf32>`。
- **​`tensor::CollapseShapeOp`​**​  **/**  **​`tensor::ExpandShapeOp`​**

  - 用于改变张量的**秩（Rank）** ，即维度数量。
  - `CollapseShapeOp`​ 可以将 `[2, 3]`​ 的维度压扁成 `[6]`。
  - `ExpandShapeOp`​ 可以将 `[6]`​ 展开成 `[2, 3]`。
  - 在处理多维卷积内存排布转换时（例如合并 Batch 维和 Channel 维），这两个 Op 极为常用。
- **​`tensor::PadOp`​**

  - 用于对张量执行**填充（Padding）** 操作。例如，在进行卷积操作前，如果需要在外围补零，就可以使用它。该 Op 允许指定每个维度前后需要填充多少，以及填充的具体数值。
- **​`tensor::SplatOp`​**​  **/**  **​`tensor::GenerateOp`​**

  - **​`SplatOp`​**：用单一的标量值填充出一个张量。例如，创建一个所有元素都是 0 的张量。
  - **​`GenerateOp`​**​：允许使用一个闭包（Region/Block 内部的逻辑），通过张量的坐标（如 `i, j, k`）动态计算出每个元素的值。
- **​`tensor::ExtractOp`​**

  - **作用**：按下标（索引）从张量中**提取出一个标量元素**，返回这个元素的值。例如 `tensor<4x8xf32>`​ 里取出 `[i][j]`​ 位置的那个元素。它需要 `indices` 参数指定每个维度上的下标。
  - **与** **​`ExtractSliceOp`​**​ **的区别**：`ExtractSliceOp`​ 提取的是**一个子张量**（Slice/Tile，形状可以是多维的）；而 `ExtractOp`​ 只提取**一个标量元素**（结果没有张量维度）。
- **​`tensor::InsertOp`​**

  - **作用**：把**一个标量值插入**到张量的指定下标位置，返回更新后的**新张量**（值语义，原张量不变）。同样需要 `indices` 指定位置。
  - **典型用法**：常与 `ExtractOp` 配合，在循环里逐个元素地读写某个张量（例如手写逐元素 kernel 时做 gather/scatter）。
  - **与** **​`InsertSliceOp`​**​ **的区别**：`InsertSliceOp`​ 插入的是一个子张量（Slice），而 `InsertOp` 只插入一个标量元素。

‍
