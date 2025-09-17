use std::path::Path;
use windows::core::{IUnknown, PCWSTR};
use windows::Win32::Foundation::E_INVALIDARG;
use windows::Win32::Media::MediaFoundation::{MFShutdown, MFStartup, MF_VERSION};
use windows::Win32::System::Com::Urlmon::E_PENDING;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::Memory::HeapEnableTerminationOnCorruption;
use windows::Win32::System::Memory::HeapSetInformation;

#[derive(Debug)]
pub struct EncoderConfig {
    pub audio_profile: usize,
    pub video_profile: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            audio_profile: 0,
            video_profile: 0,
        }
    }
}

pub fn encode_media_file<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    config: EncoderConfig,
) -> windows::core::Result<()> {
    // Convert paths to wide strings for Windows API
    let input_wide: Vec<u16> = input_path
        .as_ref()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let output_wide: Vec<u16> = output_path
        .as_ref()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();

    unsafe {
        // Enable heap termination on corruption
        HeapSetInformation(None, HeapEnableTerminationOnCorruption, None, 0)?;

        // Initialize COM
        CoInitializeEx(None, COINIT_APARTMENTTHREADED).unwrap();

        // Use drop guard to ensure CoUninitialize is called
        struct CoUninitializeGuard;
        impl Drop for CoUninitializeGuard {
            fn drop(&mut self) {
                unsafe {
                    CoUninitialize();
                }
            }
        }
        let _co_uninit = CoUninitializeGuard;

        // Initialize Media Foundation
        MFStartup(MF_VERSION, 0)?;

        // Use drop guard to ensure MFShutdown is called
        struct MFShutdownGuard;
        impl Drop for MFShutdownGuard {
            fn drop(&mut self) {
                unsafe {
                    MFShutdown().ok();
                }
            }
        }
        let _mf_shutdown = MFShutdownGuard;

        // Call the encode function (to be implemented)
        encode_file(
            PCWSTR::from_raw(input_wide.as_ptr()),
            PCWSTR::from_raw(output_wide.as_ptr()),
            &config,
        )?;

        Ok(())
    }
}

use std::time::Duration;
use windows::core::Result;
use windows::Win32::Media::MediaFoundation::*;

pub fn encode_file(input: PCWSTR, output: PCWSTR, config: &EncoderConfig) -> Result<()> {
    // Create all our COM objects up front so we can use ? operator
    let source = create_media_source(input)?;
    let duration = get_source_duration(&source)?;
    let profile = create_transcode_profile(&config)?;

    // Create the topology
    let topology = unsafe {
        // let mut topology = None;
        let topology = MFCreateTranscodeTopology(&source, output, &profile)?;
        topology
    };

    // Create and start the encoding session
    let session = Session::create()?;
    session.start_encoding_session(&topology)?;

    // Run the encoding session
    run_encoding_session(&session, duration)?;

    // Shutdown the source
    // Note: Other COM objects are automatically cleaned up when dropped
    unsafe {
        source.Shutdown()?;
    }

    Ok(())
}

// Helper function to create the media source
// use windows::core::{ComPtr, Interface, Result, PCWSTR};
use windows::core::Interface;
// use windows::Win32::System::Com::IUnknown;

fn create_media_source(url: PCWSTR) -> Result<IMFMediaSource> {
    unsafe {
        // Create the source resolver
        let resolver: IMFSourceResolver = {
            // let mut resolver = None;
            let resolver = MFCreateSourceResolver();
            resolver.unwrap()
        };

        // Use the source resolver to create the media source
        let (object_type, source): (MF_OBJECT_TYPE, IUnknown) = {
            let mut object_type = MF_OBJECT_INVALID;
            let mut source = None;
            resolver.CreateObjectFromURL(
                url,
                MF_RESOLUTION_MEDIASOURCE.0.try_into().unwrap(),
                None, // No property store
                &mut object_type,
                &mut source,
            )?;
            (object_type, source.unwrap())
        };

        // Query for the IMFMediaSource interface
        // Note: IID_PPV_ARGS is handled automatically by the windows-rs crate
        let media_source: IMFMediaSource = source.cast()?;

        Ok(media_source)
    }
}

// Helper function to get source duration
// use std::time::Duration;
// use windows::core::Result;
// use windows::Win32::Media::MediaFoundation::*;

fn get_source_duration(source: &IMFMediaSource) -> Result<Duration> {
    unsafe {
        // Create the presentation descriptor
        let pd: IMFPresentationDescriptor = source.CreatePresentationDescriptor()?;

        // raw pointer
        let x = MF_PD_DURATION;
        let raw = &x as *const windows::core::GUID;

        // Get the duration as a UINT64 (in 100-nanosecond units)
        let duration_100ns: u64 = pd.GetUINT64(raw)?;

        // Convert from 100-nanosecond units to Duration
        // Note: 1 second = 10,000,000 units (100ns each)
        let duration = if duration_100ns > 0 {
            Duration::new(
                duration_100ns / 10_000_000,                  // seconds
                ((duration_100ns % 10_000_000) * 100) as u32, // nanoseconds
            )
        } else {
            Duration::default()
        };

        Ok(duration)
    }
}

// Helper function to create transcode profile
// use windows::core::{Interface, Result, GUID};
// use windows::Win32::Media::MediaFoundation::*;
// use windows::Win32::System::Com::IUnknown;

fn create_transcode_profile(config: &EncoderConfig) -> Result<IMFTranscodeProfile> {
    unsafe {
        // Create the transcode profile
        let profile: IMFTranscodeProfile = {
            // let mut profile = None;
            let profile = MFCreateTranscodeProfile()?;
            profile
        };

        // Create and set audio attributes
        let audio_attrs = create_aac_profile(config.audio_profile)?;
        profile.SetAudioAttributes(&audio_attrs)?;

        // Create and set video attributes
        let video_attrs = create_h264_profile(config.video_profile)?;
        profile.SetVideoAttributes(&video_attrs)?;

        // Create and set container attributes
        let container_attrs: IMFAttributes = {
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 1)?;
            attrs.unwrap()
        };

        container_attrs.SetGUID(&MF_TRANSCODE_CONTAINERTYPE, &MFTranscodeContainerType_MPEG4)?;

        profile.SetContainerAttributes(&container_attrs)?;

        Ok(profile)
    }
}

// use windows::core::{Result, GUID};
// use windows::Win32::Media::MediaFoundation::*;

use super::profiles::{AAC_PROFILES, H264_PROFILES};
use super::session::Session;

fn create_h264_profile(profile_index: usize) -> Result<IMFAttributes> {
    // Ensure the profile index is valid
    if profile_index >= H264_PROFILES.len() {
        return Err(windows::core::Error::new::<&str>(
            E_INVALIDARG,
            "Invalid profile index".into(),
        ));
    }

    let profile = &H264_PROFILES[profile_index];

    unsafe {
        // Create attributes store
        let attributes: IMFAttributes = {
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 5)?;
            attrs.unwrap()
        };

        // Set the video subtype to H264
        attributes.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)?;

        // Set the H264 profile
        attributes.SetUINT32(&MF_MT_MPEG2_PROFILE, profile.profile.try_into().unwrap())?;

        // Set the frame size (packed as UINT64)
        let frame_size =
            ((profile.frame_size.Numerator as u64) << 32) | (profile.frame_size.Denominator as u64);
        attributes.SetUINT64(&MF_MT_FRAME_SIZE, frame_size)?;

        // Set the frame rate (packed as UINT64)
        let frame_rate = ((profile.fps.Numerator as u64) << 32) | (profile.fps.Denominator as u64);
        attributes.SetUINT64(&MF_MT_FRAME_RATE, frame_rate)?;

        // Set the bitrate
        attributes.SetUINT32(&MF_MT_AVG_BITRATE, profile.bitrate)?;

        Ok(attributes)
    }
}

fn create_aac_profile(profile_index: usize) -> Result<IMFAttributes> {
    // Ensure the profile index is valid
    if profile_index >= AAC_PROFILES.len() {
        return Err(windows::core::Error::new::<&str>(
            E_INVALIDARG,
            "Invalid profile index".into(),
        ));
    }

    let profile = &AAC_PROFILES[profile_index];

    unsafe {
        // Create attributes store
        let attributes: IMFAttributes = {
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 7)?;
            attrs.unwrap()
        };

        // Set the audio subtype to AAC
        attributes.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC)?;

        // Set audio attributes
        attributes.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, profile.samples_per_sec)?;
        attributes.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, profile.bits_per_sample)?;
        attributes.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, profile.num_channels)?;
        attributes.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, profile.bytes_per_sec)?;

        // Set block alignment to 1 as required
        attributes.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, 1)?;

        // Set AAC profile level (optional)
        attributes.SetUINT32(
            &MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION,
            profile.aac_profile,
        )?;

        Ok(attributes)
    }
}

// Helper function to run the encoding session
use std::io::Write;

const WAIT_PERIOD_MS: u32 = 500;
const UPDATE_INCREMENT: i64 = 5;

fn run_encoding_session(session: &Session, total_duration: Duration) -> Result<()> {
    let total_duration = total_duration.as_nanos() as i64;
    let mut previous_percent = 0;

    loop {
        match session.wait(WAIT_PERIOD_MS) {
            // E_PENDING means encoding is still in progress
            Err(e) if e.code() == E_PENDING => {
                // Get current position
                let current_position = session.get_encoding_position()?;

                // Calculate progress percentage
                let percent = (100 * current_position) / total_duration;

                // Update progress if we've moved forward enough
                if percent >= previous_percent + UPDATE_INCREMENT {
                    print!("{}%.. ", percent);
                    std::io::stdout().flush().expect("Failed to flush stdout");
                    previous_percent = percent;
                }
            }
            // Any other result means we're done (success or error)
            result => {
                println!();
                return result;
            }
        }
    }
}
