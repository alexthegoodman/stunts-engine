use windows::Win32::Media::MediaFoundation::*;

#[derive(Debug, Clone)]
pub struct H264ProfileInfo {
    // pub profile: u32,
    pub profile: i32,
    pub fps: MFRatio,
    pub frame_size: MFRatio,
    pub bitrate: u32,
}

// #[derive(Debug, Clone)]
// pub struct MFRatio {
//     pub Numerator: u32,
//     pub Denominator: u32,
// }

#[derive(Debug, Clone)]
pub struct AACProfileInfo {
    pub samples_per_sec: u32,
    pub num_channels: u32,
    pub bits_per_sample: u32,
    pub bytes_per_sec: u32,
    pub aac_profile: u32,
}

// Define constant profiles
pub const H264_PROFILES: &[H264ProfileInfo] = &[
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Base.0,
        fps: MFRatio {
            Numerator: 15,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 176,
            Denominator: 144,
        },
        bitrate: 128_000,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Base.0,
        fps: MFRatio {
            Numerator: 15,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 352,
            Denominator: 288,
        },
        bitrate: 384_000,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Base.0,
        fps: MFRatio {
            Numerator: 30,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 352,
            Denominator: 288,
        },
        bitrate: 384_000,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Base.0,
        fps: MFRatio {
            Numerator: 29970,
            Denominator: 1000,
        },
        frame_size: MFRatio {
            Numerator: 320,
            Denominator: 240,
        },
        bitrate: 528_560,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Base.0,
        fps: MFRatio {
            Numerator: 15,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 720,
            Denominator: 576,
        },
        bitrate: 4_000_000,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Main.0,
        fps: MFRatio {
            Numerator: 25,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 720,
            Denominator: 576,
        },
        bitrate: 10_000_000,
    },
    H264ProfileInfo {
        profile: eAVEncH264VProfile_Main.0,
        fps: MFRatio {
            Numerator: 30,
            Denominator: 1,
        },
        frame_size: MFRatio {
            Numerator: 352,
            Denominator: 288,
        },
        bitrate: 10_000_000,
    },
];

pub const AAC_PROFILES: &[AACProfileInfo] = &[
    AACProfileInfo {
        samples_per_sec: 96_000,
        num_channels: 2,
        bits_per_sample: 16,
        bytes_per_sec: 24_000,
        aac_profile: 0x29,
    },
    AACProfileInfo {
        samples_per_sec: 48_000,
        num_channels: 2,
        bits_per_sample: 16,
        bytes_per_sec: 24_000,
        aac_profile: 0x29,
    },
    AACProfileInfo {
        samples_per_sec: 44_100,
        num_channels: 2,
        bits_per_sample: 16,
        bytes_per_sec: 16_000,
        aac_profile: 0x29,
    },
    AACProfileInfo {
        samples_per_sec: 44_100,
        num_channels: 2,
        bits_per_sample: 16,
        bytes_per_sec: 12_000,
        aac_profile: 0x29,
    },
];
