/// A single keypoint with (x, y) coordinates and confidence score.
#[derive(Debug, Clone, Copy)]
pub struct OP18KeyPoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// A set of 133+1=134 keypoints (OpenPose 18 format) with bounding box.
///
/// The keypoint layout follows the OpenPose 18 convention with 18 body keypoints,
/// 3 left foot, 3 right foot, 68 face, 21 left hand, and 21 right hand keypoints.
///
/// Indices:
/// - 0..=17: Body (nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Reye, Leye, Rear, Lear)
/// - 18..=20: Left foot (big toe, small toe, heel)
/// - 21..=23: Right foot (big toe, small toe, heel)
/// - 24..=91: Face (68 points)
/// - 92..=112: Left hand (21 points)
/// - 113..=133: Right hand (21 points)
#[derive(Debug, Clone)]
pub struct OP18KeyPointSet {
    /// All 134 keypoints
    pub keypoints: Vec<OP18KeyPoint>,
    /// Bounding box [x1, y1, x2, y2]
    pub bbox: [f32; 4],
}

impl OP18KeyPointSet {
    /// Body keypoints (indices 0..=17, 18 points)
    pub fn body(&self) -> &[OP18KeyPoint] {
        &self.keypoints[0..18]
    }

    /// Left foot keypoints (indices 18..=20, 3 points)
    pub fn left_foot(&self) -> &[OP18KeyPoint] {
        &self.keypoints[18..21]
    }

    /// Right foot keypoints (indices 21..=23, 3 points)
    pub fn right_foot(&self) -> &[OP18KeyPoint] {
        &self.keypoints[21..24]
    }

    /// Face keypoints (indices 24..=91, 68 points)
    pub fn face(&self) -> &[OP18KeyPoint] {
        &self.keypoints[24..92]
    }

    /// Left hand keypoints (indices 92..=112, 21 points)
    pub fn left_hand(&self) -> &[OP18KeyPoint] {
        &self.keypoints[92..113]
    }

    /// Right hand keypoints (indices 113..=133, 21 points)
    pub fn right_hand(&self) -> &[OP18KeyPoint] {
        &self.keypoints[113..134]
    }

    pub fn scale(&mut self, factor: f32) {
        for kp in &mut self.keypoints {
            kp.x *= factor;
            kp.y *= factor;
        }
    }
}
