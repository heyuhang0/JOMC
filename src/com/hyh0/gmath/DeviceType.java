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
     * CL_DEVICE_TYPE_DEFAULT. This type can be used for creating a context on
     * the default device, a single device can never have this type.
     */
    DEFAULT(CLDevice.Type.DEFAULT),
    /**
     * CL_DEVICE_TYPE_ALL. This type can be used for creating a context on
     * all devices, a single device can never have this type.
     */
    ALL(CLDevice.Type.ALL);
    
    /**
     * Value of wrapped OpenCL device type.
     */
    public final CLDevice.Type TYPE;

    private DeviceType(CLDevice.Type type) {
        this.TYPE = type;
    }
}
