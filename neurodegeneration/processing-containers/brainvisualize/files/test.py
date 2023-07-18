# import vtk

# # Create a renderer and render window
# renderer = vtk.vtkRenderer()
# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)

# # Set up the render window interactor
# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(render_window)

# # Define the viewport positions and sizes
# viewport_positions = [
#     (0, 0.5, 0.5, 1),
#     (0.5, 0.5, 1, 1),
#     (0, 0, 0.5, 0.5),
#     (0.5, 0, 1, 0.5),
#     (0, 0.5, 0.5, 1),
#     (0.5, 0.5, 1, 1)
# ]

# # Create the viewports and add them to the renderer
# for position in viewport_positions:
#     viewport = vtk.vtkViewport()
#     viewport.SetViewport(*position)
#     renderer.AddViewport(viewport)

# # Add some content to the viewports (e.g., a cube)
# cube_source = vtk.vtkCubeSource()
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(cube_source.GetOutputPort())

# for i, position in enumerate(viewport_positions):
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     renderer.GetViewports().GetItem(i).AddActor(actor)

# # Set up camera and interactor
# renderer.ResetCamera()
# interactor.Initialize()
# interactor.Start()


##-------------
# #!/usr/bin/env python

# # noinspection PyUnresolvedReferences
# import vtkmodules.vtkInteractionStyle
# # noinspection PyUnresolvedReferences
# import vtkmodules.vtkRenderingOpenGL2
# from vtkmodules.vtkCommonColor import vtkNamedColors
# from vtkmodules.vtkFiltersCore import vtkContourFilter
# from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
# from vtkmodules.vtkIOImage import vtkMetaImageReader
# from vtkmodules.vtkIOImage import vtkNIFTIImageReader
# from vtkmodules.vtkImagingCore import vtkExtractVOI
# from vtkmodules.vtkRenderingCore import (
#     vtkActor,
#     vtkPolyDataMapper,
#     vtkRenderWindow,
#     vtkRenderWindowInteractor,
#     vtkRenderer
# )


# def main():
#     fileName = get_program_parameters()

#     colors = vtkNamedColors()

#     # Create the RenderWindow, Renderer and Interactor.
#     #

#     ren1 = vtkRenderer()

#     renWin = vtkRenderWindow()
#     renWin.AddRenderer(ren1)

#     iren = vtkRenderWindowInteractor()
#     iren.SetRenderWindow(renWin)

#     # Create the pipeline.
#     #

#     #reader = vtkMetaImageReader()
#     reader = vtkNIFTIImageReader()
#     reader.SetFileName(fileName)
#     reader.Update()

#     extractVOI = vtkExtractVOI()
#     extractVOI.SetInputConnection(reader.GetOutputPort())
#     extractVOI.SetVOI(0, 255, 0, 255, 45, 45)

#     iso = vtkContourFilter()
#     iso.SetInputConnection(extractVOI.GetOutputPort())
#     iso.GenerateValues(12, 500, 1150)

#     isoMapper = vtkPolyDataMapper()
#     isoMapper.SetInputConnection(iso.GetOutputPort())
#     isoMapper.ScalarVisibilityOff()

#     isoActor = vtkActor()
#     isoActor.SetMapper(isoMapper)
#     isoActor.GetProperty().SetColor(colors.GetColor3d('Wheat'))

#     outline = vtkOutlineFilter()
#     outline.SetInputConnection(extractVOI.GetOutputPort())

#     outlineMapper = vtkPolyDataMapper()
#     outlineMapper.SetInputConnection(outline.GetOutputPort())

#     outlineActor = vtkActor()
#     outlineActor.SetMapper(outlineMapper)

#     # Add the actors to the renderer, set the background and size.
#     #
#     ren1.AddActor(outlineActor)
#     ren1.AddActor(isoActor)
#     ren1.SetBackground(colors.GetColor3d('SlateGray'))
#     ren1.ResetCamera()
#     ren1.GetActiveCamera().Dolly(1.5)
#     ren1.ResetCameraClippingRange()

#     renWin.SetSize(640, 640)
#     renWin.SetWindowName('HeadSlice')

#     renWin.Render()
#     iren.Start()


# def get_program_parameters():
#     import argparse
#     description = 'Marching squares are used to generate contour lines.'
#     epilogue = '''
#     '''
#     parser = argparse.ArgumentParser(description=description, epilog=epilogue,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument('filename', help='FullHead.mhd.')
#     args = parser.parse_args()
#     return args.filename


# if __name__ == '__main__':
#     main()

##-----
# import vtk

# # Create a vtkLookupTable
# lookup_table = vtk.vtkLookupTable()

# # Set the range of scalar values
# lookup_table.SetTableRange(0.18, 1.0)

# # Set the colors for specific scalar values
# scalar_value_to_color = {
#     0.18: (1.0, 0.0, 0.0),  # Red color for value 0.18
#     0.25: (0.0, 0.0, 1.0),  # Blue color for value 0.25
#     0.35: (0.0, 1.0, 0.0),  # Green color for value 0.35
#     0.5: (0.6, 0.4, 0.2),  # Brown color for value 0.5
#     0.75: (1.0, 1.0, 0.0),  # Yellow color for value 0.75
#     1.0: (0.0, 1.0, 1.0)  # Cyan color for value 1.0
# }

# for scalar_value, color in scalar_value_to_color.items():
#     index = lookup_table.GetIndex(scalar_value)
#     lookup_table.SetTableValue(index, color[0], color[1], color[2])

# # Set the number of table values (256 for full resolution)
# lookup_table.SetNumberOfTableValues(256)

# # Build the lookup table
# lookup_table.Build()

# # Print the table values for verification
# for i in range(256):
#     scalar_value = lookup_table.GetTableValue(i)[0]
#     color = lookup_table.GetTableValue(i)[1:]
#     print(f"Scalar value: {scalar_value}, RGB color: {color[0]}, {color[1]}, {color[2]}")

# # Example usage: Get the RGB color for a specific scalar value
# scalar_value = 0.75
# index = lookup_table.GetIndex(scalar_value)
# color = lookup_table.GetTableValue(index)

# print(f"Color for scalar value {scalar_value}: {color[0]}, {color[1]}, {color[2]}")
###----
import vtk

# Create a vtkLookupTable
lookup_table = vtk.vtkLookupTable()
lookup_table.SetNumberOfTableValues(6)

# Define the color values for the specified scalar values
scalar_value_to_color = {
    0.18: (1.0, 0.0, 0.0),  # Red color for value 0.18
    0.25: (0.0, 0.0, 1.0),  # Blue color for value 0.25
    0.35: (0.0, 1.0, 0.0),  # Green color for value 0.35
    0.5: (0.6, 0.4, 0.2),   # Brown color for value 0.5
    0.75: (1.0, 1.0, 0.0),  # Yellow color for value 0.75
    1.0: (0.0, 1.0, 1.0)    # Cyan color for value 1.0
}

# Set the color values for each index in the lookup table
for index, scalar_value in enumerate(scalar_value_to_color.keys()):
    print(scalar_value)
    color = scalar_value_to_color[scalar_value]
    lookup_table.SetTableValue(index, color[0], color[1], color[2])

# Set the range of scalar values
lookup_table.SetTableRange(0.18, 1.0)

# Set the scalar visibility range
# lookup_table.SetScalarVisibility(1)

# Build the lookup table
# lookup_table.Build()
lookup_table.SetIndexedLookup(1)

# Print the table values for verification
for index in range(lookup_table.GetNumberOfTableValues()):
    print('index: ', index)
    scalar_value = lookup_table.GetTableValue(index)[0]
    color = lookup_table.GetTableValue(index)[1:]
    print(f"Index: {index}, Scalar value: {scalar_value}, RGB color: {color[0]}, {color[1]}, {color[2]}")
