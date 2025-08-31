## [Triton kernel 原语](https://triton-lang.org/main/python-api/triton.language.html)覆盖情况

| 原语                           | 类型    | 功能                     | 示例                                      | 首次出现  |
| ---------------------------- | ----- | ---------------------- | --------------------------------------- | ----- |
| `tl.arange`                  | 索引    | 生成连续索引向量               | `offsets = tl.arange(0,16)`             | 向量加算子 |
| `tl.program_id`              | 线程/程序 | 获取 kernel program ID   | `pid = tl.program_id(0)`                | 向量加算子 |
| `tl.load`                    | 内存    | 读取 GPU 全局/共享内存，可配 mask | `a = tl.load(a_ptr+offsets, mask=mask)` | 向量加算子 |
| `tl.store`                   | 内存    | 写入 GPU 内存，可配 mask      | `tl.store(c_ptr+offsets, c, mask=mask)` | 向量加算子 |
| `tl.constexpr`               | 编译常量  | 声明 kernel 参数常量         | `BLOCK_SIZE: tl.constexpr`              | 待书写   |
| `tl.cat`                     | 创建    | 拼接多个 tensor            | `tl.cat([a,b],dim=0)`                   | 待书写   |
| `tl.full`                    | 创建    | 创建指定值 tensor           | `tl.full((2,2),1.0)`                    | 待书写   |
| `tl.zeros`                   | 创建    | 全零 tensor              | `tl.zeros((2,2))`                       | 待书写   |
| `tl.zeros_like`              | 创建    | 与 tensor 同形全零          | `tl.zeros_like(a)`                      | 待书写   |
| `tl.cast`                    | 类型    | 转换 tensor 数据类型         | `tl.cast(a,tl.float32)`                 | 待书写   |
| `tl.broadcast`               | 形状    | 广播 tensor 至兼容形状        | `tl.broadcast(a,b)`                     | 待书写   |
| `tl.broadcast_to`            | 形状    | 广播到指定形状                | `tl.broadcast_to(a,(2,2))`              | 待书写   |
| `tl.expand_dims`             | 形状    | 指定维度插入长度1维度            | `tl.expand_dims(a,0)`                   | 待书写   |
| `tl.interleave`              | 形状    | 最后维度交错两个 tensor        | `tl.interleave(a,b)`                    | 待书写   |
| `tl.join`                    | 形状    | 新维度连接 tensor           | `tl.join([a,b],dim=0)`                  | 待书写   |
| `tl.permute`                 | 形状    | 重排维度                   | `tl.permute(a,(1,0))`                   | 待书写   |
| `tl.ravel`                   | 形状    | 扁平化 tensor             | `tl.ravel(a)`                           | 待书写   |
| `tl.reshape`                 | 形状    | 改变 tensor 形状           | `tl.reshape(a,(2,2))`                   | 待书写   |
| `tl.split`                   | 形状    | 最后维度分割                 | `tl.split(a,2)`                         | 待书写   |
| `tl.trans`                   | 形状    | 转置维度                   | `tl.trans(a)`                           | 待书写   |
| `tl.view`                    | 形状    | 返回不同形状视图               | `tl.view(a,(2,2))`                      | 待书写   |
| `tl.dot`                     | 线性代数  | 矩阵乘积                   | `tl.dot(a,b)`                           | 待书写   |
| `tl.dot_scaled`              | 线性代数  | 矩阵乘积，支持缩放              | `tl.dot_scaled(a,b)`                    | 待书写   |
| `tl.make_tensor_descriptor`  | 内存/指针 | 创建 tensor 描述符          | `tl.make_tensor_descriptor(a)`          | 待书写   |
| `tl.load_tensor_descriptor`  | 内存/指针 | 从描述符加载数据               | `tl.load_tensor_descriptor(a_desc)`     | 待书写   |
| `tl.store_tensor_descriptor` | 内存/指针 | 存储数据到描述符               | `tl.store_tensor_descriptor(c_desc,c)`  | 待书写   |
| `tl.make_block_ptr`          | 内存/指针 | 指向 tensor 块指针          | `tl.make_block_ptr(a,(0,0))`            | 待书写   |
| `tl.advance`                 | 内存/指针 | 指针偏移                   | `tl.advance(a_ptr,1)`                   | 待书写   |
| `tl.flip`                    | 索引    | 指定维度翻转                 | `tl.flip(a,0)`                          | 待书写   |
| `tl.where`                   | 索引    | 条件选择元素                 | `tl.where(mask,a,b)`                    | 待书写   |
| `tl.swizzle2d`               | 索引    | 2D 索引行列互换              | `tl.swizzle2d(a)`                       | 待书写   |
| `tl.abs`                     | 数学    | 元素绝对值                  | `tl.abs(a)`                             | 待书写   |
| `tl.cdiv`                    | 数学    | 元素上取整除法                | `tl.cdiv(a,b)`                          | 待书写   |
| `tl.ceil`                    | 数学    | 上取整                    | `tl.ceil(a)`                            | 待书写   |
| `tl.clamp`                   | 数学    | 限定元素范围                 | `tl.clamp(a,0,1)`                       | 待书写   |
| `tl.cos`                     | 数学    | 元素余弦                   | `tl.cos(a)`                             | 待书写   |
| `tl.div_rn`                  | 数学    | 精确除法四舍五入               | `tl.div_rn(a,b)`                        | 待书写   |
| `tl.erf`                     | 数学    | 误差函数                   | `tl.erf(a)`                             | 待书写   |
| `tl.exp`                     | 数学    | 指数                     | `tl.exp(a)`                             | 待书写   |
| `tl.exp2`                    | 数学    | 2 为底指数                 | `tl.exp2(a)`                            | 待书写   |
| `tl.fdiv`                    | 数学    | 快速除法                   | `tl.fdiv(a,b)`                          | 待书写   |
| `tl.floor`                   | 数学    | 下取整                    | `tl.floor(a)`                           | 待书写   |
| `tl.fma`                     | 数学    | 乘加运算                   | `tl.fma(a,b,c)`                         | 待书写   |
| `tl.log`                     | 数学    | 自然对数                   | `tl.log(a)`                             | 待书写   |
