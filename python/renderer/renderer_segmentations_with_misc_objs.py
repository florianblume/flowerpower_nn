# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
import struct
from vispy import app, gloo
import OpenGL.GL as gl

# WARNING: doesn't work with Qt4 (update() does not call on_draw()??)
app.use_app('PyGlet') # Set backend

# Segmentation vertex shader
# This shader simply renders the segmentation mask for the given object
#-------------------------------------------------------------------------------
_segmentation_vertex_code = """
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec3 a_color;
varying vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = vec4(a_color, 1.0);
}
"""

# Fragment shader that simply applies the color output by the vertex shader
#-------------------------------------------------------------------------------
_segmentation_fragment_code = """
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
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

#-------------------------------------------------------------------------------
class _Canvas(app.Canvas):
    def __init__(self, model_to_render, misc_models, size, K, clip_near, clip_far,
                 bg_color=(0.0, 0.0, 0.0, 0.0), ambient_weight=0.1, segmentation_color=None):
        """
        mode is from ['rgb', 'depth', 'rgb+depth']
        """
        app.Canvas.__init__(self, show=False, size=size)

        #gloo.gl.use_gl('gl2 debug')

        self.size = size
        self.shape = (self.size[1], self.size[0])
        self.bg_color = bg_color
        self.ambient_weight = ambient_weight

        self.segmentation = np.array([])
        self.segmentation_color = segmentation_color

        self.model_to_render = model_to_render
        self.misc_models = misc_models

        # Model matrix
        self.mat_model = np.eye(4, dtype=np.float32) # From object space to world space

        # Projection matrix
        self.mat_proj = _compute_calib_proj(K, 0, 0, size[0], size[1], clip_near, clip_far)

        self.Rs = [model_to_render['R']]
        self.ts = [model_to_render['t']]

        # Create buffers
        self.vertex_buffers = [gloo.VertexBuffer(model_to_render['obj']['vertices'])]
        self.index_buffers = [gloo.IndexBuffer(model_to_render['obj']['faces'].flatten().astype(np.uint32))]

        for model in misc_models:
            self.vertex_buffers.append(gloo.VertexBuffer(model['obj']['vertices']))
            self.index_buffers.append(gloo.IndexBuffer(model['obj']['faces'].flatten().astype(np.uint32)))
            self.Rs.append(model['R'])
            self.ts.append(model['t'])

        # We manually draw the hidden canvas
        self.update()

    def on_draw(self, event):
        self.draw_segmentation()
        app.quit()

    def draw_segmentation(self):
        program = gloo.Program(_segmentation_vertex_code, _segmentation_fragment_code)

        # Texture where we render the scene
        render_tex = gloo.Texture2D(shape=self.shape + (4,))

        # Frame buffer object
        fbo = gloo.FrameBuffer(render_tex, gloo.RenderBuffer(self.shape))
        with fbo:
            gloo.set_state(depth_test=True)
            gloo.set_state(cull_face=True)
            gloo.set_cull_face('back')  # Back-facing polygons will be culled
            gloo.set_clear_color(self.bg_color)
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *self.size)

            for index in range(len(self.vertex_buffers)):
                # View matrix (transforming also the coordinate system from OpenCV to
                # OpenGL camera space)
                self.mat_view = np.eye(4, dtype=np.float32) # From world space to eye space
                self.mat_view[:3, :3], self.mat_view[:3, 3] = self.Rs[index], self.ts[index].squeeze()
                yz_flip = np.eye(4, dtype=np.float32)
                yz_flip[1, 1], yz_flip[2, 2] = -1, -1
                self.mat_view = yz_flip.dot(self.mat_view) # OpenCV to OpenGL camera system
                self.mat_view = self.mat_view.T # OpenGL expects column-wise matrix format
                program['u_mvp'] = _compute_model_view_proj(self.mat_model, self.mat_view, self.mat_proj)

                program.bind(self.vertex_buffers[index])

                if index == 0:
                    # The first object is drawn with the segmentation color
                    program['a_color'] = self.segmentation_color
                else:
                    program['a_color'] = [0, 0, 0]
                program.draw('triangles', self.index_buffers[index])

            # Retrieve the contents of the FBO texture
            self.segmentation = gloo.read_pixels((0, 0, self.size[0], self.size[1]))[:, :, :3]
            self.segmentation = np.copy(self.segmentation)

def render(model_to_render, misc_models, im_size, K, clip_near=100, clip_far=2000,
           surf_color=None, bg_color=(0.0, 0.0, 0.0, 0.0),
           ambient_weight=0.1,
           segmentation_color=None):
    

    # Process input data
    #---------------------------------------------------------------------------
    # Make sure vertices and faces are provided in the model
    assert({'pts', 'faces'}.issubset(set(model_to_render['obj'].keys())))

    for model in misc_models:
        assert({'pts', 'faces'}.issubset(set(model['obj'].keys())))

    # If we don't incorporate the color the rendering doesn't work...
    colors = np.ones((model_to_render['obj']['pts'].shape[0], 4), np.float32) * 0.5
    vertices_type = [('a_position', np.float32, 3),
                     #('a_normal', np.float32, 3),
                     ('a_color', np.float32, colors.shape[1])]
    zipped = zip(model_to_render['obj']['pts'], colors)
    vertices = np.array(list(zipped), vertices_type)
    model_to_render['obj']['vertices'] = vertices

    for i in range(len(misc_models)):
        model = misc_models[i]
        colors = np.ones((model['obj']['pts'].shape[0], 4), np.float32) * 0.5
        vertices_type = [('a_position', np.float32, 3),
                     #('a_normal', np.float32, 3),
                     ('a_color', np.float32, colors.shape[1])]
        zipped = zip(model['obj']['pts'], colors)
        vertices = np.array(list(zipped), vertices_type)
        model['obj']['vertices'] = vertices
        misc_models[i] = model

    c = _Canvas(model_to_render, misc_models, im_size, K, clip_near, clip_far,
                bg_color, ambient_weight, segmentation_color)
    app.run()

    #---------------------------------------------------------------------------
    out = c.segmentation

    c.close()
    return out