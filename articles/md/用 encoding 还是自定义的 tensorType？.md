# 用 encoding 还是自定义的 tensorType？

强烈建议使用 encoding！！！！！可以最大限度的兼容mlir的生态！！！！

‍

## TensorEncoding 基础概念

```cpp
// 🎯 TensorEncoding 是附加在 TensorType 上的属性
// tensor<shape, element_type, encoding>
tensor<1x32x16x44xf32, #my_encoding>
```

## 1. 增加 (Create) - 创建 TensorEncoding

### 方法1：在 C++ 中创建

your\_code.cpp

Apply

```cpp

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"

// 🎉 创建带 encoding 的 tensor
mlir::RankedTensorType createEncodedTensor(mlir::MLIRContext *context) {
    // 创建 encoding 属性
    auto encoding = mlir::StringAttr::get(context, "my_custom_encoding");
    
    // 创建带 encoding 的 tensor 类型
    auto shape = llvm::ArrayRef<int64_t>{1, 32, 16, 44};
    auto elementType = mlir::FloatType::getF32(context);
    
    return mlir::RankedTensorType::get(shape, elementType, encoding);
}

// 🎯 使用自定义 encoding 属性
mlir::RankedTensorType createHalEncodedTensor(mlir::MLIRContext *context) {
    // 假设你有自定义的 HalEncodingAttr
    auto halEncoding = tbc::hals::HalEncodingAttr::get(context,
        MemorySpace::DDR,
        StorageLayout::NCHW,
        0,
        TensorKind::IOTensor);
    
    auto shape = llvm::ArrayRef<int64_t>{1, 32, 16, 44};
    auto elementType = mlir::FloatType::getF32(context);
    
    return mlir::RankedTensorType::get(shape, elementType, halEncoding);
}
```

### 方法2：在 Operation 中创建

SomeOp.cpp

Apply

```cpp

void SomeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value input, mlir::Attribute encoding) {
    // 🎯 获取输入类型
    auto inputType = input.getType().cast<mlir::RankedTensorType>();
    
    // 🎉 创建带新 encoding 的输出类型
    auto outputType = mlir::RankedTensorType::get(
        inputType.getShape(),
        inputType.getElementType(),
        encoding  // 新的 encoding
    );
    
    state.addOperands(input);
    state.addTypes(outputType);
}
```

### 方法3：在 Pattern 中创建

SomePattern.cpp

Apply

```cpp

class AddEncodingPattern : public mlir::OpRewritePattern<SomeOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(SomeOp op, mlir::PatternRewriter &rewriter) const override {
        auto inputType = op.getInput().getType().cast<mlir::RankedTensorType>();
        
        // 🎉 创建新的 encoding
        auto newEncoding = mlir::StringAttr::get(getContext(), "optimized_layout");
        
        // 🎯 创建新的类型
        auto newType = mlir::RankedTensorType::get(
            inputType.getShape(),
            inputType.getElementType(),
            newEncoding
        );
        
        // 替换操作
        auto newOp = rewriter.create<SomeOp>(op.getLoc(), newType, op.getInput());
        rewriter.replaceOp(op, newOp);
        
        return success();
    }
};
```

## 2. 删除 (Delete) - 移除 TensorEncoding

### 方法1：移除 encoding

RemoveEncoding.cpp

Apply

```cpp

mlir::RankedTensorType removeEncoding(mlir::RankedTensorType tensorType) {
    // 🎯 创建没有 encoding 的新类型
    return mlir::RankedTensorType::get(
        tensorType.getShape(),
        tensorType.getElementType()
        // 不传 encoding 参数，默认为 nullptr
    );
}

// 🎉 在 Pattern 中移除 encoding
class RemoveEncodingPattern : public mlir::OpRewritePattern<SomeOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(SomeOp op, mlir::PatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType().cast<mlir::RankedTensorType>();
        
        // 🚨 检查是否有 encoding
        if (!resultType.getEncoding()) {
            return failure(); // 没有 encoding，不需要处理
        }
        
        // 🎯 移除 encoding
        auto newType = mlir::RankedTensorType::get(
            resultType.getShape(),
            resultType.getElementType()
        );
        
        auto newOp = rewriter.create<SomeOp>(op.getLoc(), newType, op.getOperands());
        rewriter.replaceOp(op, newOp);
        
        return success();
    }
};
```

### 方法2：条件性移除

ConditionalRemove.cpp

Apply

```cpp

class ConditionalRemoveEncodingPattern : public mlir::OpRewritePattern<SomeOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(SomeOp op, mlir::PatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType().cast<mlir::RankedTensorType>();
        auto encoding = resultType.getEncoding();
        
        if (!encoding) return failure();
        
        // 🎯 检查特定类型的 encoding
        if (auto halEncoding = encoding.dyn_cast<tbc::hals::HalEncodingAttr>()) {
            // 只移除 DDR 内存空间的 encoding
            if (halEncoding.getMemorySpace() == MemorySpace::DDR) {
                auto newType = mlir::RankedTensorType::get(
                    resultType.getShape(),
                    resultType.getElementType()
                );
                
                auto newOp = rewriter.create<SomeOp>(op.getLoc(), newType, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        }
        
        return failure();
    }
};
```

## 3. 修改 (Update) - 更新 TensorEncoding

### 方法1：替换 encoding

UpdateEncoding.cpp

Apply

```cpp

class UpdateEncodingPattern : public mlir::OpRewritePattern<SomeOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    
    mlir::LogicalResult matchAndRewrite(SomeOp op, mlir::PatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType().cast<mlir::RankedTensorType>();
        auto oldEncoding = resultType.getEncoding();
        
        // 🎯 创建新的 encoding
        mlir::Attribute newEncoding;
        
        if (auto halEncoding = oldEncoding.dyn_cast_or_null<tbc::hals::HalEncodingAttr>()) {
            // 🎉 更新 HalEncoding 的某些字段
            newEncoding = tbc::hals::HalEncodingAttr::get(
                getContext(),
                MemorySpace::SRAM,  // 改变内存空间
                halEncoding.getLayout(),  // 保持布局
                halEncoding.getAddr(),    // 保持地址
                TensorKind::TempTensor    // 改变类型
            );
        } else {
            // 🎯 添加新的 encoding
            newEncoding = mlir::StringAttr::get(getContext(), "new_encoding");
        }
        
        // 🎉 创建新类型
        auto newType = mlir::RankedTensorType::get(
            resultType.getShape(),
            resultType.getElementType(),
            newEncoding
        );
        
        auto newOp = rewriter.create<SomeOp>(op.getLoc(), newType, op.getOperands());
        rewriter.replaceOp(op, newOp);
        
        return success();
    }
};
```

### 方法2：增量更新

IncrementalUpdate.cpp

Apply

```cpp

// 🎯 工具函数：更新 HalEncoding 的内存空间
mlir::RankedTensorType updateMemorySpace(mlir::RankedTensorType tensorType, 
                                        MemorySpace newMemorySpace,
                                        mlir::MLIRContext *context) {
    auto encoding = tensorType.getEncoding();
    
    if (auto halEncoding = encoding.dyn_cast_or_null<tbc::hals::HalEncodingAttr>()) {
        // 🎉 只更新内存空间，其他保持不变
        auto newEncoding = tbc::hals::HalEncodingAttr::get(
            context,
            newMemorySpace,                    // 新的内存空间
            halEncoding.getLayout(),          // 保持原有布局
            halEncoding.getAddr(),            // 保持原有地址
            halEncoding.getKind()             // 保持原有类型
        );
        
        return mlir::RankedTensorType::get(
            tensorType.getShape(),
            tensorType.getElementType(),
            newEncoding
        );
    }
    
    return tensorType; // 如果不是 HalEncoding，返回原类型
}

// 🎯 工具函数：更新布局
mlir::RankedTensorType updateLayout(mlir::RankedTensorType tensorType,
                                   StorageLayout newLayout,
                                   mlir::MLIRContext *context) {
    auto encoding = tensorType.getEncoding();
    
    if (auto halEncoding = encoding.dyn_cast_or_null<tbc::hals::HalEncodingAttr>()) {
        auto newEncoding = tbc::hals::HalEncodingAttr::get(
            context,
            halEncoding.getMemorySpace(),     // 保持原有内存空间
            newLayout,                        // 新的布局
            halEncoding.getAddr(),            // 保持原有地址
            halEncoding.getKind()             // 保持原有类型
        );
        
        return mlir::RankedTensorType::get(
```
