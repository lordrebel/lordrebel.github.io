# mlir td定义中 的特殊变量

## TableGen 特殊变量完全指南 📚

### 1. OpInterface 中的特殊变量

`cur_op->getNumResults()`

#### \$\_op - 当前操作的指针

示例 Interface 定义

Apply

```tablegen

def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
    InterfaceMethod<
      "void", "doSomething", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // $_op 是指向当前 Op $_op->getOperation()
        auto operands = $_op.getOperands();  // 获取操作数
        auto results = $_op.getResults();    // 获取结果
        auto attrs = $_op.getAttrs();        // 获取属性
        
        // 可以调用 Operation 的任何方法
        if ($_op->hasAttr("some_attr")) {
          // ...
        }
      }]
    >
  ];
}
```

**生成的 C++ 代码：**

```cpp
// $_op 会被替换为 static_cast<ConcreteOp*>(this)
template <typename ConcreteOp>
struct MyInterfaceTrait {
    void doSomething() {
        auto operands = static_cast<ConcreteOp*>(this)->getOperands();
        auto results = static_cast<ConcreteOp*>(this)->getResults();
        // ...
    }
};
```

#### \$\_self - 当前对象（较少使用）

```tablegen
InterfaceMethod<
  "bool", "isValid", (ins),
  /*defaultImplementation=*/[{
    // $_self 通常用于 Attribute/Type Interface
    return $_self.isa<SomeType>();
  }]
>
```

---

### 2. Op 定义中的特殊变量

#### \$\<operand\_name\> - 访问操作数

示例 Op 定义

Apply

```tablegen

def MyAddOp : Op<MyDialect, "add"> {
  let arguments = (ins 
    AnyTensor:$lhs,      // 左操作数
    AnyTensor:$rhs,      // 右操作数
    I64Attr:$scale       // 属性
  );
  let results = (outs AnyTensor:$output);
  
  let extraClassDeclaration = [{
    // 在 C++ 代码中可以这样用：
    Value getLhs() { return getOperand(0); }  // 或者用生成的 getLhs()
    Value getRhs() { return getOperand(1); }  // 或者用生成的 getRhs()
    int64_t getScale() { return getScaleAttr().getInt(); }
  }];
  
  let hasVerifier = 1;
}
```

**在 Verifier 中使用：**

MyAddOp.cpp

Apply

```cpp

LogicalResult MyAddOp::verify() {
    // 可以直接用生成的 getter
    auto lhs = getLhs();
    auto rhs = getRhs();
    auto scale = getScale();
    
    // 验证逻辑
    if (scale <= 0) {
        return emitError("scale must be positive");
    }
    return success();
}
```

---

### 3. Pattern Rewrite 中的特殊变量

#### \$\_builder - PatternRewriter 引用

示例 Pattern 定义

Apply

```tablegen

def FoldAddZero : Pat<
  (MyAddOp $lhs, (ConstantOp $zero), $scale),  // 匹配模式
  (replaceWithValue $lhs),                      // 替换模式
  [(IsZero $zero)]                              // 约束条件
>;

// 更复杂的例子：
def FuseAddMul : Pat<
  (MyMulOp (MyAddOp $x, $y), $z),
  (MyFusedOp $x, $y, $z),
  [],
  // 可以添加额外的 C++ 代码
  [{
    // $_builder 是 PatternRewriter&
    auto loc = $_builder.getUnknownLoc();
    auto newOp = $_builder.create<MyFusedOp>(loc, ...);
    return newOp;
  }]
>;
```

#### \$\_loc - Location 对象

```tablegen
def CreateNewOp : NativeCodeCall<
  "$_builder.create<NewOp>($_loc, $0, $1)"
>;

// 使用：
def MyPattern : Pat<
  (OldOp $a, $b),
  (CreateNewOp $a, $b)  // $_loc 会自动传入
>;
```

#### \$0, \$1, \$2, ... - 匹配的参数

示例 Native Code Call

Apply

```tablegen

def CreateScaledAdd : NativeCodeCall<
  "$_builder.create<AddOp>($_loc, $0, $1, $_builder.getI64IntegerAttr($2 * 2))"
>;

def ScalePattern : Pat<
  (MyOp $input1, $input2, $scale),
  (CreateScaledAdd $input1, $input2, $scale)
  // $0 = $input1, $1 = $input2, $2 = $scale
>;
```

---

### 4. Attribute/Type 定义中的特殊变量

#### \$\_self - 当前 Attribute/Type

示例 Attribute 定义

Apply

```tablegen

def MyAttr : AttrDef<MyDialect, "MyAttr"> {
  let parameters = (ins "int64_t":$value);
  
  let extraClassDeclaration = [{
    bool isPositive() const {
      // $_self 在这里不适用，直接用 getValue()
      return getValue() > 0;
    }
  }];
}

// Type Interface 示例
def ShapedTypeInterface : TypeInterface<"ShapedType"> {
  let methods = [
    InterfaceMethod<
      "bool", "hasRank", (ins),
      /*defaultImplementation=*/[{
        // $_self 指向当前 Type
        return $_self.isa<RankedTensorType>();
      }]
    >
  ];
}
```

---

### 5. Pass 定义中的特殊变量

#### \$\_op - 当前处理的 Operation

示例 Pass 定义

Apply

```tablegen

def MyPass : Pass<"my-pass", "func::FuncOp"> {
  let summary = "My optimization pass";
  let constructor = "createMyPass()";
  
  // 在 Pass 实现中：
  let dependentDialects = ["arith::ArithDialect"];
}
```

**在 Pass 实现中：**

MyPass.cpp

Apply

```cpp

struct MyPass : public MyPassBase<MyPass> {
  void runOnOperation() override {
    // getOperation() 返回当前处理的 Operation
    auto funcOp = getOperation();  // 这里是 func::FuncOp
    
    funcOp.walk([&](Operation* op) {
      // 遍历所有操作
      if (auto addOp = dyn_cast<MyAddOp>(op)) {
        // 处理 AddOp
      }
    });
  }
};
```

---

### 6. 约束条件中的特殊变量

#### \$\_self - 被约束的值

示例约束定义

Apply

```tablegen

def IsPositive : Constraint<
  CPred<"$_self.cast<IntegerAttr>().getInt() > 0">,
  "attribute must be positive"
>;

def MyOp : Op<MyDialect, "my_op"> {
  let arguments = (ins 
    I64Attr:$value
  );
  
  // 使用约束
  let hasVerifier = 1;
}

// 在 Pattern 中使用：
def OptimizePositive : Pat<
  (MyOp $val),
  (NewOp $val),
  [(IsPositive $val)]  // $val 会被传给 $_self
>;
```

---

## 完整示例：综合运用 🎨

MyInterface.td

Apply

```tablegen

#ifndef MY_INTERFACE
#define MY_INTERFACE

include "mlir/IR/OpBase.td"

// ============ Interface 定义 ============
def MyInterface : OpInterface<"MyInterface"> {
  let description = "示例接口";
  let cppNamespace = "::aice";
  
  let methods = [
    InterfaceMethod<
      "LogicalResult", "optimize",
      (ins "PatternRewriter&":$rewriter),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // $_op 指向当前 Operation
        auto loc = $_op->getLoc();
        auto operands = $_op->getOperands();
        
        // 使用 rewriter 创建新操作
        auto newOp = rewriter.create<SomeOp>(loc, operands);
        rewriter.replaceOp($_op, newOp);
        
        return success();
      }]
    >,
    
    InterfaceMethod<
      "bool", "canOptimize", (ins),
      /*defaultImplementation=*/[{
        // 检查操作是否可以优化
        return $_op->getNumOperands() > 0 && 
               $_op->getNumResults() == 1;
      }]
    >
  ];
}

// ============ Op 定义 ============
def MyComputeOp : Op<MyDialect, "compute", [MyInterface]> {
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    I64Attr:$scale,
    BoolAttr:$fuse_relu
  );
  let results = (outs AnyTensor:$output);
  
  // 使用 $input, $weight 等访问操作数
  let extraClassDeclaration = [{
    Value getInput() { return getOperand(0); }
    Value getWeight() { return getOperand(1); }
    int64_t getScale() { return getScaleAttr().getInt(); }
  }];
}

// ============ Pattern 定义 ============
def FuseReluPattern : Pat<
  // 匹配：Compute -> Relu
  (ReluOp (MyComputeOp $input, $weight, $scale, ConstBoolAttrFalse)),
  // 替换：Compute with fuse_relu=true
  (MyComputeOp $input, $weight, $scale, ConstBoolAttrTrue),
  // 约束
  [(HasOneUse $0)]  // $0 是 MyComputeOp 的结果
>;

// ============ Native Code Call ============
def CreateFusedOp : NativeCodeCall<
  "$_builder.create<MyComputeOp>("
    "$_loc, "           // 位置
    "$0, "              // input
    "$1, "              // weight  
    "$_builder.getI64IntegerAttr($2 * 2), "  // scale * 2
    "$_builder.getBoolAttr(true)"            // fuse_relu
  ")"
>;

#endif
```

|变量|使用场景|含义|示例|
| ------| --------------------------------------| -----------------------| -------------------|
|`$_op`|OpInterface, Pattern|当前 Operation 指针|`$_op->getOperands()`|
|`$_self`|Attribute/Type Interface, Constraint|当前对象本身|`$_self.isa<IntegerType>()`|
|`$_builder`|Pattern, NativeCodeCall|PatternRewriter 引用|`$_builder.create<NewOp>(...)`|
|`$_loc`|Pattern, NativeCodeCall|Location 对象|`$_builder.create<Op>($_loc, ...)`|
|`$_ctx`|较少使用|MLIRContext 指针|`$_ctx->getI64Type()`|
|`$<name>`|Op 定义|访问命名的操作数/属性|`$input`​, `$scale`|
|`$0, $1, $2...`|Pattern, NativeCodeCall|匹配的参数（按位置）|`$0` = 第一个参数|
