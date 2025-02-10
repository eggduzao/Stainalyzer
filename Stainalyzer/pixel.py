
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_hls, hls_to_rgb

class Pixel:
    """
    Represents a pixel with spatial coordinates and multiple color representations (RGB, LAB, HSV, HSL, CMYK, XYZ, LUV).

    Attributes:
        closure (bool): Whether all color attributes must remain synchronized.
        Coordinates:
            x (float): X-coordinate. None indicates 0D or 1D space.
            y (float): Y-coordinate. None indicates 1D or 2D space.
            z (float): Z-coordinate. None indicates 2D space.
        Colors:
            color_name (str): Name of the pixel's color.
            RGB:
                red_rgb (uint8): Red channel.
                green_rgb (uint8): Green channel.
                blue_rgb (uint8): Blue channel.
            LAB:
                lightness_lab (float): Lightness channel.
                a_lab (float): A channel.
                b_lab (float): B channel.
            HSV:
                hue_hsv (float): Hue channel.
                saturation_hsv (float): Saturation channel.
                value_hsv (float): Value channel.
            HSL:
                hue_hsl (float): Hue channel.
                saturation_hsl (float): Saturation channel.
                lightness_hsl (float): Lightness channel.
            CMYK:
                cyan_cymk (float): Cyan channel.
                yellow_cymk (float): Yellow channel.
                magenta_cymk (float): Magenta channel.
                black_cymk (float): Black channel.
            XYZ:
                x_xyz (float): X channel.
                y_xyz (float): Y channel.
                z_xyz (float): Z channel.
            LUV:
                l_luv (float): L channel.
                u_luv (float): U channel.
                v_luv (float): V channel.
    """

    def __init__(
        self,
        closure=True,
        x=None,
        y=None,
        z=None,
        color_name="",
        red_rgb=0,
        green_rgb=0,
        blue_rgb=0,
        lightness_lab=0.0,
        a_lab=0.0,
        b_lab=0.0,
        hue_hsv=0.0,
        saturation_hsv=0.0,
        value_hsv=0.0,
        hue_hsl=0.0,
        saturation_hsl=0.0,
        lightness_hsl=0.0,
        cyan_cymk=0.0,
        yellow_cymk=0.0,
        magenta_cymk=0.0,
        black_cymk=0.0,
        x_xyz=0.0,
        y_xyz=0.0,
        z_xyz=0.0,
        l_luv=0.0,
        u_luv=0.0,
        v_luv=0.0,
    ):
        # Closure attribute
        self.closure = np.bool_(closure)

        # Coordinates
        self.x = np.float32(x) if x is not None else None
        self.y = np.float32(y) if y is not None else None
        self.z = np.float32(z) if z is not None else None

        # Color name
        self.color_name = color_name

        # RGB
        self.red_rgb = np.uint8(red_rgb)
        self.green_rgb = np.uint8(green_rgb)
        self.blue_rgb = np.uint8(blue_rgb)

        # LAB
        self.lightness_lab = np.float32(lightness_lab)
        self.a_lab = np.float32(a_lab)
        self.b_lab = np.float32(b_lab)

        # HSV
        self.hue_hsv = np.float32(hue_hsv)
        self.saturation_hsv = np.float32(saturation_hsv)
        self.value_hsv = np.float32(value_hsv)

        # HSL
        self.hue_hsl = np.float32(hue_hsl)
        self.saturation_hsl = np.float32(saturation_hsl)
        self.lightness_hsl = np.float32(lightness_hsl)

        # CMYK
        self.cyan_cymk = np.float32(cyan_cymk)
        self.yellow_cymk = np.float32(yellow_cymk)
        self.magenta_cymk = np.float32(magenta_cymk)
        self.black_cymk = np.float32(black_cymk)

        # XYZ
        self.x_xyz = np.float32(x_xyz)
        self.y_xyz = np.float32(y_xyz)
        self.z_xyz = np.float32(z_xyz)

        # LUV
        self.l_luv = np.float32(l_luv)
        self.u_luv = np.float32(u_luv)
        self.v_luv = np.float32(v_luv)

        # Synchronize colors if closure is enabled
        if self.closure:
            self.sync_colors()

    def sync_colors(self):
        """
        Synchronize all color representations based on RGB values.
        """
        self._update_hsv_from_rgb()
        self._update_hsl_from_rgb()
        self._update_lab_from_rgb()
        self._update_cymk_from_rgb()
        self._update_xyz_from_rgb()
        self._update_luv_from_rgb()

    def _update_hsv_from_rgb(self):
        """
        Update HSV (Hue, Saturation, Value) values based on the current RGB values.

        Converts RGB values (0–255) to HSV using the standard transformation. 
        Updates hue_hsv, saturation_hsv, and value_hsv attributes.
        """
        r, g, b = (
            self.red_rgb / 255.0,
            self.green_rgb / 255.0,
            self.blue_rgb / 255.0,
        )
        h, s, v = rgb_to_hsv(r, g, b)
        self.hue_hsv = h * 360  # Scale hue to degrees
        self.saturation_hsv = s * 100  # Scale saturation to percentage
        self.value_hsv = v * 100  # Scale value to percentage

    def _update_hsl_from_rgb(self):
        """
        Update HSL (Hue, Saturation, Lightness) values based on the current RGB values.

        Converts RGB values (0–255) to HSL using the standard transformation. 
        Updates hue_hsl, saturation_hsl, and lightness_hsl attributes.
        """
        r, g, b = (
            self.red_rgb / 255.0,
            self.green_rgb / 255.0,
            self.blue_rgb / 255.0,
        )
        h, l, s = rgb_to_hls(r, g, b)
        self.hue_hsl = h * 360  # Scale hue to degrees
        self.saturation_hsl = s * 100  # Scale saturation to percentage
        self.lightness_hsl = l * 100  # Scale lightness to percentage

    def _update_lab_from_rgb(self):
        """
        Update LAB (Lightness, A, B) values based on the current RGB values.

        Converts RGB values (0–255) to LAB using OpenCV. 
        Updates lightness_lab, a_lab, and b_lab attributes.
        """
        rgb = np.array([[[self.red_rgb, self.green_rgb, self.blue_rgb]]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0]
        self.lightness_lab = lab[0] * (100 / 255)  # Normalize L to 0–100
        self.a_lab = lab[1] - 128  # Normalize A to -128 to +128
        self.b_lab = lab[2] - 128  # Normalize B to -128 to +128

    def _update_cymk_from_rgb(self):
        """
        Update CMYK (Cyan, Magenta, Yellow, Black) values based on the current RGB values.

        Converts RGB values (0–255) to CMYK using the standard transformation. 
        Updates cyan_cymk, magenta_cymk, yellow_cymk, and black_cymk attributes.
        """
        r, g, b = (
            self.red_rgb / 255.0,
            self.green_rgb / 255.0,
            self.blue_rgb / 255.0,
        )
        k = 1 - max(r, g, b)
        self.black_cymk = k * 100
        self.cyan_cymk = (1 - r - k) / (1 - k) * 100 if k < 1 else 0
        self.magenta_cymk = (1 - g - k) / (1 - k) * 100 if k < 1 else 0
        self.yellow_cymk = (1 - b - k) / (1 - k) * 100 if k < 1 else 0

    def _update_xyz_from_rgb(self):
        """
        Update XYZ (Tristimulus values) based on the current RGB values.

        Converts RGB values (0–255) to XYZ using the standard transformation. 
        Updates x_xyz, y_xyz, and z_xyz attributes.
        """
        r, g, b = (
            self.red_rgb / 255.0,
            self.green_rgb / 255.0,
            self.blue_rgb / 255.0,
        )

        # Convert to a linear RGB space
        r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

        # Convert to XYZ using the D65 illuminant
        self.x_xyz = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100
        self.y_xyz = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100
        self.z_xyz = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100

    def _update_luv_from_rgb(self):
        """
        Update LUV (Lightness, U, V) values based on the current RGB values.

        Converts RGB values (0–255) to LUV using OpenCV. 
        Updates l_luv, u_luv, and v_luv attributes.
        """
        rgb = np.array([[[self.red_rgb, self.green_rgb, self.blue_rgb]]], dtype=np.uint8)
        luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv)[0][0]
        self.l_luv = luv[0] * (100 / 255)  # Normalize L to 0–100
        self.u_luv = luv[1] - 128  # Normalize U to -128 to +128
        self.v_luv = luv[2] - 128  # Normalize V to -128 to +128

    @property
    def rgb_tuple(self):
        """Get the RGB tuple (R, G, B)."""
        return (self.red_rgb, self.green_rgb, self.blue_rgb)

    @rgb_tuple.setter
    def rgb_tuple(self, rgb):
        """Set RGB values (R, G, B) from a tuple.
        
        Parameters:
            rgb (tuple): A tuple of 3 integers (R, G, B) in the range 0–255.
        """
        self.red_rgb, self.green_rgb, self.blue_rgb = map(np.uint8, rgb)
        if self.closure:
            self.sync_colors()

    @property
    def lab_tuple(self):
        """Get the LAB tuple (Lightness, A, B)."""
        return (self.lightness_lab, self.a_lab, self.b_lab)

    @lab_tuple.setter
    def lab_tuple(self, lab):
        """Set LAB values (Lightness, A, B) from a tuple.
        
        Parameters:
            lab (tuple): A tuple of 3 floats (Lightness, A, B).
        """
        self.lightness_lab, self.a_lab, self.b_lab = map(np.float32, lab)
        if self.closure:
            self.sync_colors()

    @property
    def hsv_tuple(self):
        """Get the HSV tuple (Hue, Saturation, Value)."""
        return (self.hue_hsv, self.saturation_hsv, self.value_hsv)

    @hsv_tuple.setter
    def hsv_tuple(self, hsv):
        """Set HSV values (Hue, Saturation, Value) from a tuple.
        
        Parameters:
            hsv (tuple): A tuple of 3 floats (Hue, Saturation, Value).
        """
        self.hue_hsv, self.saturation_hsv, self.value_hsv = map(np.float32, hsv)
        if self.closure:
            self.sync_colors()

    @property
    def hsl_tuple(self):
        """Get the HSL tuple (Hue, Saturation, Lightness)."""
        return (self.hue_hsl, self.saturation_hsl, self.lightness_hsl)

    @hsl_tuple.setter
    def hsl_tuple(self, hsl):
        """Set HSL values (Hue, Saturation, Lightness) from a tuple.
        
        Parameters:
            hsl (tuple): A tuple of 3 floats (Hue, Saturation, Lightness).
        """
        self.hue_hsl, self.saturation_hsl, self.lightness_hsl = map(np.float32, hsl)
        if self.closure:
            self.sync_colors()

    @property
    def cymk_tuple(self):
        """Get the CYMK tuple (Cyan, Yellow, Magenta, Black)."""
        return (self.cyan_cymk, self.yellow_cymk, self.magenta_cymk, self.black_cymk)

    @cymk_tuple.setter
    def cymk_tuple(self, cymk):
        """Set CYMK values (Cyan, Yellow, Magenta, Black) from a tuple.
        
        Parameters:
            cymk (tuple): A tuple of 4 floats (Cyan, Yellow, Magenta, Black).
        """
        self.cyan_cymk, self.yellow_cymk, self.magenta_cymk, self.black_cymk = map(np.float32, cymk)
        if self.closure:
            self.sync_colors()

    @property
    def xyz_tuple(self):
        """Get the XYZ tuple (X, Y, Z)."""
        return (self.x_xyz, self.y_xyz, self.z_xyz)

    @xyz_tuple.setter
    def xyz_tuple(self, xyz):
        """Set XYZ values (X, Y, Z) from a tuple.
        
        Parameters:
            xyz (tuple): A tuple of 3 floats (X, Y, Z).
        """
        self.x_xyz, self.y_xyz, self.z_xyz = map(np.float32, xyz)
        if self.closure:
            self.sync_colors()

    @property
    def luv_tuple(self):
        """Get the LUV tuple (Lightness, U, V)."""
        return (self.l_luv, self.u_luv, self.v_luv)

    @luv_tuple.setter
    def luv_tuple(self, luv):
        """Set LUV values (Lightness, U, V) from a tuple.
        
        Parameters:
            luv (tuple): A tuple of 3 floats (Lightness, U, V).
        """
        self.l_luv, self.u_luv, self.v_luv = map(np.float32, luv)
        if self.closure:
            self.sync_colors()

    @property
    def hex_color(self):
        """
        Get the hexadecimal representation of the RGB color.

        Returns:
            str: Hexadecimal color code as a string, e.g., '#rrggbb'.
        """
        return f"#{self.red_rgb:02x}{self.green_rgb:02x}{self.blue_rgb:02x}"

    @hex_color.setter
    def hex_color(self, hex_value):
        """
        Set the RGB color using a hexadecimal value.

        Parameters:
            hex_value (str): Hexadecimal color code, e.g., '#rrggbb' or 'rrggbb'.
        """
        hex_value = hex_value.lstrip('#')
        self.red_rgb = np.uint8(int(hex_value[0:2], 16))
        self.green_rgb = np.uint8(int(hex_value[2:4], 16))
        self.blue_rgb = np.uint8(int(hex_value[4:6], 16))
        if self.closure:
            self.sync_colors()

    @property
    def color_name(self):
        """
        Get the name of the closest color to the current RGB values.

        Returns:
            str: The name of the closest color.
        """
        return self._color_name

    @color_name.setter
    def color_name(self, color_name_dictionary):
        """
        Set the name of the closest color based on the current RGB values.

        Parameters:
            color_name_dictionary (dict): Dictionary of hex-to-name mappings.
        """
        rgb_color = self.rgb_tuple
        hex_color = f"{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"

        # Exact match
        if hex_color in color_name_dictionary:
            self._color_name = color_name_dictionary[hex_color]
            return

        # Zigzag search for the closest color
        min_distance = float('inf')
        closest_name = None
        for hex_key, name in color_name_dictionary.items():
            # Convert hex_key to RGB
            r = float(int(hex_key[0:2], 16))
            g = float(int(hex_key[2:4], 16))
            b = float(int(hex_key[4:6], 16))

            # Calculate Manhattan distance
            distance = self._calculate_color_distance((r, g, b), rgb_color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        self._color_name = closest_name

    def __eq__(self, other):
        """
        Default equality check between this Pixel and another Pixel object.

        By default, compares both colors and coordinates.

        Parameters:
            other (Pixel): Another Pixel object.

        Returns:
            bool: True if both color and coordinate data match, False otherwise.
        """
        if not isinstance(other, Pixel):
            return NotImplemented
        return (
            self.rgb_tuple == other.rgb_tuple and
            self.x == other.x and
            self.y == other.y and
            self.z == other.z
        )

    def compare_colors(self, other):
        """
        Compare the colors of this Pixel with another Pixel.

        Parameters:
            other (Pixel): Another Pixel object.

        Returns:
            bool: True if colors match, False otherwise.
        """
        if not isinstance(other, Pixel):
            return NotImplemented
        return self.rgb_tuple == other.rgb_tuple

    def compare_coordinates(self, other):
        """
        Compare the coordinates of this Pixel with another Pixel.

        Parameters:
            other (Pixel): Another Pixel object.

        Returns:
            bool: True if coordinates match, False otherwise.
        """
        if not isinstance(other, Pixel):
            return NotImplemented
        return (
            self.x == other.x and
            self.y == other.y and
            self.z == other.z
        )

    def __str__(self):
        """
        Return a string representation of the Pixel object.

        This method provides a human-readable summary of the Pixel object's attributes,
        including its color name, coordinates, and primary color information in RGB, LAB, 
        and HSV color spaces.

        Returns:
            str: A string representation of the Pixel object, formatted as:
                 "Pixel(Color Name: <color_name>, Coordinates: (<x>, <y>, <z>), 
                 RGB: (<red>, <green>, <blue>), LAB: (<lightness>, <a>, <b>), 
                 HSV: (<hue>, <saturation>, <value>))"
        """
        color_info = f"RGB: {self.rgb_tuple}, LAB: {self.lab_tuple}, HSV: {self.hsv_tuple}"
        coord_info = f"Coordinates: ({self.x}, {self.y}, {self.z})"
        name_info = f"Color Name: {self.color_name or 'Unnamed'}"
        return f"Pixel({name_info}, {coord_info}, {color_info})"

    def to_dict(self):
        """
        Serialize the pixel data to a dictionary.

        Returns:
            dict: A dictionary representation of the pixel.
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rgb": self.rgb_tuple,
            "lab": self.lab_tuple,
            "hsv": self.hsv_tuple,
            "color_name": self.color_name
        }

    def _calculate_color_distance(self, color1, color2):
        """
        Calculate the Manhattan distance between two colors.

        Parameters:
            color1 (tuple): RGB values of the first color.
            color2 (tuple): RGB values of the second color.

        Returns:
            float: The Manhattan distance between the colors.
        """
        return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))


    def distance_to(self, other, space='rgb'):
        """
        Calculate the distance to another Pixel object.

        Parameters:
            other (Pixel): The other Pixel object.
            space (str): The color space or coordinate system ('rgb', 'lab', 'xyz', or 'spatial').

        Returns:
            float: The computed distance.
        """
        if space == 'spatial':
            if self.x is not None and other.x is not None:
                return sqrt(
                    (self.x - other.x) ** 2 +
                    (self.y - other.y) ** 2 +
                    (self.z - other.z) ** 2
                )
            raise ValueError("Spatial coordinates are missing in one or both pixels.")
        elif space == 'rgb':
            return self._calculate_color_distance(self.rgb_tuple, other.rgb_tuple)
        elif space == 'lab':
            return self._calculate_color_distance(
                (self.lightness_lab, self.a_lab, self.b_lab),
                (other.lightness_lab, other.a_lab, other.b_lab)
            )
        elif space == 'xyz':
            return self._calculate_color_distance(
                (self.x_xyz, self.y_xyz, self.z_xyz),
                (other.x_xyz, other.y_xyz, other.z_xyz)
            )
        else:
            raise ValueError("Unsupported space. Use 'rgb', 'lab', 'xyz', or 'spatial'.")

    def interpolate_with(self, other, factor=0.5, space='rgb'):
        """
        Interpolate between the current pixel and another pixel.

        Parameters:
            other (Pixel): The other Pixel object.
            factor (float): Interpolation factor (0 = current pixel, 1 = other pixel).
            space (str): The color space for interpolation ('rgb', 'lab', etc.).

        Returns:
            Pixel: A new Pixel object with interpolated values.
        """
        factor = np.clip(factor, 0, 1)  # Ensure factor is within [0, 1]
        if space == 'rgb':
            interpolated_rgb = tuple(
                np.uint8((1 - factor) * c1 + factor * c2)
                for c1, c2 in zip(self.rgb_tuple, other.rgb_tuple)
            )
            return Pixel(red_rgb=interpolated_rgb[0], green_rgb=interpolated_rgb[1], blue_rgb=interpolated_rgb[2])
        elif space == 'lab':
            interpolated_lab = tuple(
                (1 - factor) * c1 + factor * c2
                for c1, c2 in zip(
                    (self.lightness_lab, self.a_lab, self.b_lab),
                    (other.lightness_lab, other.a_lab, other.b_lab)
                )
            )
            return Pixel(lightness_lab=interpolated_lab[0], a_lab=interpolated_lab[1], b_lab=interpolated_lab[2])
        else:
            raise ValueError("Unsupported space. Use 'rgb' or 'lab'.")

    def apply_color_filter(self, filter_type='brightness', factor=1.2):
        """
        Apply a color filter to the current pixel.

        Parameters:
            filter_type (str): The type of filter ('brightness', 'contrast', 'sepia').
            factor (float): The intensity of the filter.

        Returns:
            None
        """
        if filter_type == 'brightness':
            self.red_rgb = np.clip(self.red_rgb * factor, 0, 255).astype(np.uint8)
            self.green_rgb = np.clip(self.green_rgb * factor, 0, 255).astype(np.uint8)
            self.blue_rgb = np.clip(self.blue_rgb * factor, 0, 255).astype(np.uint8)
            if self.closure:
                self.sync_colors()
        elif filter_type == 'sepia':
            r, g, b = self.rgb_tuple
            self.red_rgb = np.uint8(np.clip(0.393 * r + 0.769 * g + 0.189 * b, 0, 255))
            self.green_rgb = np.uint8(np.clip(0.349 * r + 0.686 * g + 0.168 * b, 0, 255))
            self.blue_rgb = np.uint8(np.clip(0.272 * r + 0.534 * g + 0.131 * b, 0, 255))
            if self.closure:
                self.sync_colors()
        else:
            raise ValueError("Unsupported filter type. Use 'brightness', 'contrast', or 'sepia'.")
