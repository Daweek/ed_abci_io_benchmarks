#version 430

in vec2 vert;
in vec2 pixcolor;
out vec3 colorf;

uniform int rtype;
layout(binding=3) uniform all_uniforms
{
    vec2 rnd;
    int mappings;
    float rot_l;
    float rot_r;
};

layout(binding=0) uniform perspective_block
{
    vec4 bufproj;
};

mat4 orthogonalPerspective(float left, float right, float bottom, float top, float near, float far) {
  mat4 result = mat4(1.0);

  result[0][0] = 2.0 / (right - left);
  result[1][1] = 2.0 / (top - bottom);
  result[2][2] = -2.0 / (far - near);
  result[3][0] = -(right + left) / (right - left);
  result[3][1] = -(top + bottom) / (top - bottom);
  result[3][2] = -(far + near) / (far - near);

  return result;
}

void main() {

    mat4 shader_projection;

    float alfa = 0.015;
     
    float distx = abs(bufproj.x - bufproj.z);
    float disty = abs(bufproj.y - bufproj.w);

    float r = bufproj.x + (distx * alfa) ;
    float t = bufproj.y + (disty * alfa) ;
    float l = bufproj.z - (distx * alfa) ;
    float b = bufproj.w - (disty * alfa) ;

    // float r = bufproj.x ;
    // float t = bufproj.y ;
    // float l = bufproj.z ;
    // float b = bufproj.w ;

    float n = -0.1;
    float f = 100.0;

    shader_projection = orthogonalPerspective(l,r,b,t,n,f);

    vec4 original = shader_projection * vec4(vert, 0.0, 1.0);
    gl_Position = original * vec4(rot_l,rot_r,0.0,1.0);
    colorf = vec3(pixcolor,0.5);
}