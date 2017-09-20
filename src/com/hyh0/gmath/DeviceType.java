package com.hyh0.gmath;

import com.jogamp.opencl.CLDevice;

public enum DeviceType {
    /**
     * CL_DEVICE_TYPE_CPU
     */
    CPU(CLDevice.Type.CPU),
    /**
     * CL_DEVICE_TYPE_GPU
     */
    GPU(CLDevice.Type.GPU),
    /**
     * CL_DEVICE_TYPE_DEFAULT. Use the device with largest flops
     */
    DEFAULT(null);
    
    /**
     * Value of wrapped OpenCL device type.
     */
    public final CLDevice.Type TYPE;

    private DeviceType(CLDevice.Type type) {
        this.TYPE = type;
    }
}
