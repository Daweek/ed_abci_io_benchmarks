#version 430

#define NTHREADS 256
layout (local_size_x = NTHREADS) in;

struct mp
{
	float a, b, c, d; // scaling/rotation matrix
	float x, y; // translation vertex
	float p; // mapping probability
};

layout(binding=3) uniform all_uniforms
{
    vec2 rnd;
    int mappings;
    float rot_l;
    float rot_r;
};

layout(binding=0) uniform map_block
{
    mp map[8];
};

layout(std430, binding=0) buffer in_vec
{
    vec2 d_poss[];
};

layout(std430, binding=1) buffer in_color
{
    vec2 d_color[];
};

uniform int numPoints;
shared mp s_map[8];

float rand(vec2 co)
{
    float a = 12.9898*(rnd.x*1);
    float b = 78.233*(rnd.y*1);
    float c = 43758.5453;
    float dt= dot(co.xy ,vec2(a,b));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

void main() {
    // Test to compute shader
    // blockIdx.x * blockDim.x + threadIdx.x;
    uint index = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint stride = gl_WorkGroupSize.x * gl_NumWorkGroups.x;

    float limit = 1.69999e+6;  

    for (int i=0; i<mappings ; i++ ){
        s_map[i] = map[i];
    }
    memoryBarrierShared();
    barrier();

    uint currentTarget = index % mappings;
    vec2 currentPoss;
    vec2 newPoss;

    currentPoss.x = s_map[currentTarget].x;
    currentPoss.y = s_map[currentTarget].y;

    for(uint i = index; i<numPoints; i+= stride){

        d_color[i] = vec2(0.5,0.5);
        // d_color[i].x =  i / numPoints;
        // d_color[i].y = currentTarget;

        float prob = rand(vec2(index,currentPoss.x));
        float cump = 0.0;

        for (int j=0; j < mappings; j++){
            cump += s_map[j].p;
            if(prob < cump){
                currentTarget = j;
                break;
            }
        }

        newPoss.x =     s_map[currentTarget].a * currentPoss.x +
                        s_map[currentTarget].b * currentPoss.y +
                        s_map[currentTarget].x;

        newPoss.y =     s_map[currentTarget].c * currentPoss.x +
                        s_map[currentTarget].d * currentPoss.y +
                        s_map[currentTarget].y;

        if(newPoss.x > -limit && newPoss.x < limit ){
            currentPoss.x = newPoss.x;
            // d_color[i] = vec2(1.0,1.0);
        }
        else{
            // d_color[i] = vec2(1.0,0.5);
            newPoss.x = 0.0;
            currentPoss.x = newPoss.x;
        }

        if(newPoss.y > -limit && newPoss.y < limit ){
            currentPoss.y = newPoss.y;
            // d_color[i] = vec2(1.0,1.0);
        }
        else{
            // d_color[i] = vec2(1.0,0.5);
            newPoss.y = 0.0;
            currentPoss.y = newPoss.y;
        }

        d_poss[i].x = newPoss.x;
        d_poss[i].y = newPoss.y;
    }
    
}
    