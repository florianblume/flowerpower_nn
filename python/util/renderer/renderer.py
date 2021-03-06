# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
import struct
from vispy import app, gloo
import OpenGL.GL as gl

AVAILABLE_RENDERING_MODES = ['rgb', 'depth', 'obj_coords', 'segmentation', 'bounding_boxes']

# WARNING: doesn't work with Qt4 (update() does not call on_draw()??)
app.use_app('PyGlet') # Set backend

# Segmentation vertex shader
# This shader simply renders the segmentation mask for the given object
#-------------------------------------------------------------------------------
_segmentation_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec3 a_color;
varying vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# Fragment shader that simply applies the color output by the vertex shader
#-------------------------------------------------------------------------------
_segmentation_fragment_code = """
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"""

# Object coordinates representation vertex shader
# This shader code renders the 3D object coordinates of the surface of the object
# into the buffer. This shader is only supposed to visualize the object coordinates
# and should not be used to retrieve the actual coordinates, as the precision of
# the color buffer is too small (max number is 255, but coordinates can be e.g. 500)
#-------------------------------------------------------------------------------
_obj_coords_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec4 a_color;
varying float x;
varying float y;
varying float z;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    // We simply store the location of the vertex
    // Pixels between vertices will be interpolated automatically
    x = a_position.x;
    y = a_position.y;
    z = a_position.z;
}
"""

# Fragment shader that simply applies the color output by the vertex shader
#-------------------------------------------------------------------------------
_obj_coords_fragment_code = """
varying float x;
varying float y;
varying float z;
void main() {
    // We just put the coordinates as colors, OpenGL allows floating point colors
    // Only the format of the Texture we render to has to be set to float
    gl_FragColor = vec4(x, y, z, 1.0);
}
"""

# Color vertex shader
#-------------------------------------------------------------------------------
_bounding_box_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;
attribute vec3 a_position;
attribute vec4 a_color;
varying vec4 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
}
"""

# Color fragment shader
#-------------------------------------------------------------------------------
_bounding_box_fragment_code = """
uniform float u_light_ambient_w;
varying vec4 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
void main() {
    // Face normal in eye coordinates
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));
    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;
    vec4 color = light_w * v_color;
    gl_FragColor = vec4(color.xyz, 1.0);
}"""

# Color vertex shader
#-------------------------------------------------------------------------------
_color_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;
attribute vec3 a_position;
attribute vec4 a_color;
varying vec4 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
}
"""

# Color fragment shader
#-------------------------------------------------------------------------------
_color_fragment_code = """
uniform float u_light_ambient_w;
varying vec4 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
void main() {
    // Face normal in eye coordinates
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));
    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;
    gl_FragColor = light_w * v_color;
}
"""

# Depth vertex shader
# Ref: https://github.com/julienr/vertex_visibility/blob/master/depth.py
#-------------------------------------------------------------------------------
# Getting the depth from the depth buffer in OpenGL is doable, see here:
#   http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
#   http://web.archive.org/web/20130426093607/http://www.songho.ca/opengl/gl_projectionmatrix.html
#   http://stackoverflow.com/a/6657284/116067
# But it is hard to get good precision, as explained in this article:
# http://dev.theomader.com/depth-precision/
#
# Once the vertex is in view space (view * model * v), its depth is simply the
# Z axis. So instead of reading from the depth buffer and undoing the projection
# matrix, we store the Z coord of each vertex in the COLOR buffer and then
# read from the color buffer. OpenGL desktop allows for float32 color buffer
# components.
_depth_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec4 a_color;
varying float v_eye_depth;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates
    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;
}
"""

# Depth fragment shader
#-------------------------------------------------------------------------------
_depth_fragment_code = """
varying float v_eye_depth;
void main() {
    gl_FragColor = vec4(v_eye_depth, 0.0, 0.0, 1.0);
}
"""

# Functions to calculate transformation matrices
# Note that OpenGL expects the matrices to be saved column-wise
# (Ref: http://www.songho.ca/opengl/gl_transform.html)
#-------------------------------------------------------------------------------
# Model-view matrix
def _compute_model_view(model, view):
    return np.dot(model, view)

# Model-view-projection matrix
def _compute_model_view_proj(model, view, proj):
    return np.dot(np.dot(model, view), proj)

# Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)
def _compute_normal_matrix(model, view):
    return np.linalg.inv(np.dot(model, view)).T

# Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix
#-------------------------------------------------------------------------------
# Ref:
# 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
# 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
def _compute_calib_proj(K, x0, y0, w, h, nc, fc, window_coords='y_down'):
    """
    :param K: Camera matrix.
    :param x0, y0: The camera image origin (normally (0, 0)).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: OpenGL projection matrix.
    """
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same
    if window_coords == 'y_up':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective

    # Draw the images right side up and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords
    else:
        assert window_coords == 'y_down'
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective
    return proj.T

def compute_view_matrix(R, t):
    # View matrix (transforming also the coordinate system from OpenCV to
    # OpenGL camera space)
    mat_view = np.eye(4, dtype=np.float32) # From world space to eye space
    mat_view[:3, :3], mat_view[:3, 3] = R, t.squeeze()
    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    mat_view = yz_flip.dot(mat_view) # OpenCV to OpenGL camera system
    mat_view = mat_view.T # OpenGL expects column-wise matrix format
    return mat_view

#-------------------------------------------------------------------------------
class _Canvas(app.Canvas):
    def __init__(self, shape, K, models, clip_near, clip_far,
                 bg_color=(0.0, 0.0, 0.0, 0.0), ambient_weight=0.1,
                 render_rgb=True, render_depth=True, 
                 render_obj_coords=True, render_segmentation=True,
                 render_bounding_boxes=True):

        app.Canvas.__init__(self, show=False, size=(shape[1], shape[0]))

        #gloo.gl.use_gl('gl2 debug')

        self.size = (shape[1], shape[0])
        self.shape = shape
        self.bg_color = bg_color
        self.ambient_weight = ambient_weight
        self.render_rgb = render_rgb
        self.render_depth = render_depth
        self.render_obj_coords = render_obj_coords
        self.render_segmentation = render_segmentation
        self.render_bounding_boxes = render_bounding_boxes
        self.models = models

        self.rgb = np.array([])
        self.depth = np.array([])
        self.obj_coords = np.array([])
        self.obj_coords_rep = np.array([])
        self.segmentation = np.array([])

        # Model matrix
        self.mat_model = np.eye(4, dtype=np.float32) # From object space to world space

        # Projection matrix
        self.mat_proj = _compute_calib_proj(K, 0, 0, self.size[0], self.size[1], clip_near, clip_far)

        self.vertex_buffers = []
        self.index_buffers = []
        self.poses = []

        for model in models:
            self.vertex_buffers.append(gloo.VertexBuffer(model['vertices']))
            self.index_buffers.append(gloo.IndexBuffer(model['faces'].flatten().astype(np.uint32)))
            self.poses.append({'R' : model['R'], 't' : model['t']})

        # We manually draw the hidden canvas
        self.update()

    def on_draw(self, event):
        if self.render_rgb:
            self.draw_color() # Render color image
        if self.render_depth:
            self.draw_depth() # Render depth image
        if self.render_obj_coords:
            self.draw_obj_coords()
        if self.render_segmentation:
            self.draw_segmentation()
        if self.render_bounding_boxes:
            self.draw_bounding_boxes()
        app.quit() # Immediately exit the application after the first drawing   

    def do_render(self, program, fbo, finish_operation):
        with fbo:
            gloo.set_state(depth_test=True)
            gloo.set_state(cull_face=True)
            gloo.set_cull_face('back')  # Back-facing polygons will be culled
            gloo.set_clear_color(self.bg_color)
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *self.size)

            for i in range(len(self.vertex_buffers)):
                self.mat_view = compute_view_matrix(self.poses[i]['R'], self.poses[i]['t'])
                program['u_mv'] = _compute_model_view(self.mat_model, self.mat_view)
                # program['u_nm'] = compute_normal_matrix(self.model, self.view)
                program['u_mvp'] = _compute_model_view_proj(self.mat_model, self.mat_view, self.mat_proj)
                program.bind(self.vertex_buffers[i])
                program.draw('triangles', self.index_buffers[i])

            if finish_operation:
                finish_operation(fbo)

    def draw_color(self):
        program = gloo.Program(_color_vertex_code, _color_fragment_code)
        program['u_light_eye_pos'] = [0, 0, 0]
        program['u_light_ambient_w'] = self.ambient_weight
        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,))
        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape))
        self.do_render(program, fbo, self.retrieve_color)

    def retrieve_color(self, fbo):
        self.rgb = gloo.read_pixels((0, 0, self.size[0], self.size[1]))[:, :, :3]
        self.rgb = np.copy(self.rgb)

    def draw_bounding_boxes(self):

        program = gloo.Program(_bounding_box_vertex_code, _bounding_box_fragment_code)

        program['u_light_eye_pos'] = [0, 0, 0]
        program['u_light_ambient_w'] = self.ambient_weight

        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,), format=gl.GL_RGBA,
                                    internalformat=gl.GL_RGBA)
        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape))
        with fbo:
            gloo.set_state(depth_test=True)
            gloo.set_state(cull_face=True)
            gloo.set_state(blend=True)
            gloo.set_state(blend_func=('src_alpha', 'one_minus_src_alpha'))
            gloo.set_cull_face('back')  # Back-facing polygons will be culled
            gloo.set_clear_color(self.bg_color)
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *self.size)

            for i in range(len(self.vertex_buffers)):
                self.mat_view = compute_view_matrix(self.poses[i]['R'], self.poses[i]['t'])
                program['u_mv'] = _compute_model_view(self.mat_model, self.mat_view)
                # program['u_nm'] = compute_normal_matrix(self.model, self.view)
                program['u_mvp'] = _compute_model_view_proj(self.mat_model, self.mat_view, self.mat_proj)
                #program.bind(self.vertex_buffers[i])
                #program.draw('triangles', self.index_buffers[i])

                model = self.models[i]
                vertices = model['vertices']
                x_max = np.amax(vertices['a_position'][:,0]) 
                x_min = np.amin(vertices['a_position'][:,0])
                y_max = np.amax(vertices['a_position'][:,1]) 
                y_min = np.amin(vertices['a_position'][:,1])
                z_max = np.amax(vertices['a_position'][:,2]) 
                z_min = np.amin(vertices['a_position'][:,2])
                line_vertices = [ # First side of the cube
                                 [x_max, y_min, z_max],
                                 [x_max, y_min, z_min],
                                 [x_min, y_min, z_min],
                                 [x_min, y_min, z_max],
                                 [x_max, y_min, z_max],
                                 # Side wall of the cube
                                 [x_max, y_max, z_max],
                                 [x_max, y_max, z_min],
                                 [x_max, y_min, z_min],
                                 [x_max, y_max, z_min],
                                 # Next side wall
                                 [x_max, y_max, z_min],
                                 [x_min, y_max, z_min],
                                 [x_min, y_min, z_min],
                                 [x_min, y_max, z_min],
                                 # Next side wall
                                 [x_min, y_max, z_max],
                                 [x_min, y_min, z_max],
                                 [x_min, y_max, z_max],

                                 [x_max, y_max, z_max]]
                program['a_position'] = gloo.VertexBuffer(line_vertices)
                program['a_color'] = gloo.VertexBuffer(np.tile(vertices['a_color'][0], [len(line_vertices), 1]))
                gl.glLineWidth(10)
                gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
                program.draw('line_strip')

            self.retrieve_bounding_boxes()

    def retrieve_bounding_boxes(self):
        self.bounding_boxes = gloo.read_pixels((0, 0, self.size[0], self.size[1]))[:, :, :3]
        self.bounding_boxes = np.copy(self.bounding_boxes)

    def draw_depth(self):
        program = gloo.Program(_depth_vertex_code, _depth_fragment_code)
        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,), format=gl.GL_RGBA,
                                    internalformat=gl.GL_RGBA32F)
        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape, format='depth'))
        self.do_render(program, fbo, self.retrieve_depth)

    def retrieve_depth(self, fbo):
        # Retrieve the contents of the FBO texture
        self.depth = self.read_fbo_color_rgba32f(fbo)
        self.depth = self.depth[:, :, 0] # Depth is saved in the first channel

    def draw_obj_coords(self):
        program = gloo.Program(_obj_coords_vertex_code, _obj_coords_fragment_code)
        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,), format=gl.GL_RGBA,
                                    internalformat=gl.GL_RGBA32F)
        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape))
        self.do_render(program, fbo, self.retrieve_obj_coords)

    def retrieve_obj_coords(self, fbo):
        # Retrieve the contents of the FBO texture
        self.obj_coords = self.read_fbo_color_rgba32f(fbo)[:, :, :3] # Last channel is alpha, we only need the first three
        self.obj_coords = np.copy(self.obj_coords)

    def draw_segmentation(self):
        program = gloo.Program(_segmentation_vertex_code, _segmentation_fragment_code)
        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,))
        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape))
        self.do_render(program, fbo, self.retrieve_segmentation)

    def retrieve_segmentation(self, fbo):
        # Retrieve the contents of the FBO texture
        self.segmentation = gloo.read_pixels((0, 0, self.size[0], self.size[1]))[:, :, :3]
        self.segmentation = np.copy(self.segmentation)

    @staticmethod
    def read_fbo_color_rgba32f(fbo):
        """
        Read the color attachment from a FBO, assuming it is GL_RGBA_32F.
        # Ref: https://github.com/julienr/vertex_visibility/blob/master/depth.py
        """
        h, w = fbo.color_buffer.shape[:2]
        x, y = 0, 0
        im = gl.glReadPixels(x, y, w, h, gl.GL_RGBA, gl.GL_FLOAT)
        im = np.frombuffer(im, np.float32)
        im.shape = h, w, 4
        im = im[::-1, :]

        return im

#-------------------------------------------------------------------------------
# Ref: https://github.com/vispy/vispy/blob/master/examples/demo/gloo/offscreen.py
def render(im_shape, K, models, surf_colors=None, 
            clip_near=10, clip_far=2000, bg_color=(0.0, 0.0, 0.0, 0.0),
           ambient_weight=0.1, modes=['rgb']):

    prepared_models = []
    for i, model in enumerate(models):
        # Process input data
        #---------------------------------------------------------------------------
        # Make sure vertices and faces are provided in the model
        assert({'pts', 'faces', 'R', 't'}.issubset(set(model.keys())))

        # Set color of vertices
        if surf_colors is None:
            if 'colors' in model.keys():
                assert(model['pts'].shape[0] == model['colors'].shape[0])
                colors = model['colors']
                if colors.max() > 1.0:
                    colors /= 255.0 # Color values are expected to be in range [0, 1]
            else:
                colors = np.ones((model['pts'].shape[0], 4), np.float32) * 0.5
        else:
            colors = np.tile(list(surf_colors[i]), [model['pts'].shape[0], 1])

        vertices_type = [('a_position', np.float32, 3),
                            #('a_normal', np.float32, 3),
                            ('a_color', np.float32, colors.shape[1])]
        zipped = zip(model['pts'], colors)
        vertices = np.array(list(zipped), vertices_type)
        prepared_models.append({'vertices' : vertices, 
                                'faces' : model['faces'],
                                'R' : model['R'],
                                't' : model['t']})

    # Rendering
    #---------------------------------------------------------------------------
    render_rgb = 'rgb' in modes
    render_depth = 'depth' in modes
    render_obj_coords = 'obj_coords' in modes
    render_segmentation = 'segmentation' in modes
    render_bounding_boxes = 'bounding_boxes' in modes
    c = _Canvas(im_shape, K, prepared_models, clip_near, clip_far,
                bg_color, ambient_weight, 
                render_rgb, render_depth, render_obj_coords, render_segmentation, render_bounding_boxes)
    app.run()

    #---------------------------------------------------------------------------
    out = {}
    if render_rgb:
        out['rgb'] = c.rgb
    if render_depth:
        out['depth'] = c.depth
    if render_obj_coords:
        out['obj_coords'] = c.obj_coords
    if render_segmentation:
        out['segmentation'] = c.segmentation
    if render_bounding_boxes:
        out['bounding_boxes'] = c.bounding_boxes
    if any(elem for elem in modes if elem not in AVAILABLE_RENDERING_MODES):
        out = None
        print('Error: Unknown rendering mode in modes: {}'.format(modes))
        exit(-1)

    c.close()
    return out