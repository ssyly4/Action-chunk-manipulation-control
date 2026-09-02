# 外部依赖说明

本目录说明不属于 NERO 控制源码的依赖。

- `pyAgxArm` 暂时由 `NERO_ARM_SDK_ROOT` 指向官方 SDK 工作区。
- OSQP、CasADi 与 TOPPRA 的 Python 依赖包各自位于对应轨迹模块旁的 `vendor/` 中，因为其中的 CPython 3.12 二进制轮子必须与实机策略环境匹配。

这些目录不包含数采数据或模型 checkpoint。
