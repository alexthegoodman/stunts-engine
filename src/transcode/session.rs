use std::sync::atomic::AtomicI32;
use windows::core::{implement, Interface, Result, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::*;
use windows::Win32::Media::KernelStreaming::GUID_NULL;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::Threading::*;
use Urlmon::E_PENDING;

#[implement(IMFAsyncCallback)]
pub struct Session {
    ref_count: AtomicI32,
    session: Option<IMFMediaSession>,
    clock: Option<IMFPresentationClock>,
    status: HRESULT,
    wait_event: HANDLE,
}

impl Session {
    pub fn create() -> Result<Session> {
        let mut session = Session {
            ref_count: AtomicI32::new(1),
            session: None,
            clock: None,
            status: S_OK,
            wait_event: HANDLE(std::ptr::null_mut()),
        };

        unsafe {
            // Create media session
            let media_session: IMFMediaSession = MFCreateMediaSession(None)?;

            // Get the clock
            let clock: IMFClock = media_session.GetClock()?;
            let presentation_clock: IMFPresentationClock = clock.cast()?;

            // Create wait event
            let wait_event = CreateEventW(None, false, false, None)?;
            if wait_event.is_invalid() {
                return Err(windows::core::Error::from_win32());
            }

            // Store everything in our struct
            session.session = Some(media_session);
            session.clock = Some(presentation_clock);
            session.wait_event = wait_event;

            // Get the COM interface pointer to our Session object for the callback
            // could this cast be right?
            let callback: IMFAsyncCallback = session.cast()?;

            // Begin getting events with our callback
            session
                .session
                .as_ref()
                .unwrap()
                .BeginGetEvent(&callback, None)?;

            Ok(session)
        }
    }

    pub fn start_encoding_session(&self, topology: &IMFTopology) -> Result<()> {
        unsafe {
            let session = self.session.as_ref().unwrap();
            session.SetTopology(0, topology)?;

            let var_start = PROPVARIANT::default();
            session.Start(&GUID_NULL, &var_start)?;

            Ok(())
        }
    }

    pub fn get_encoding_position(&self) -> Result<i64> {
        unsafe {
            let clock = self.clock.as_ref().unwrap();
            // let mut time = 0;
            let time = clock.GetTime()?;
            Ok(time)
        }
    }

    pub fn wait(&self, timeout_ms: u32) -> Result<()> {
        unsafe {
            match WaitForSingleObject(self.wait_event, timeout_ms) {
                WAIT_OBJECT_0 => {
                    if self.status.is_err() {
                        Err(windows::core::Error::from(self.status))
                    } else {
                        Ok(())
                    }
                }
                _ => Err(windows::core::Error::new::<&str>(
                    E_PENDING,
                    "Operation pending".into(),
                )),
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if let Some(session) = &self.session {
                session.Shutdown().ok();
            }
            if !self.wait_event.is_invalid() {
                CloseHandle(self.wait_event);
            }
        }
    }
}

// #[allow(non_snake_case)]
impl IMFAsyncCallback_Impl for Session_Impl {
    fn GetParameters(&self, _pdwflags: *mut u32, _pdwqueue: *mut u32) -> Result<()> {
        Err(windows::core::Error::new::<&str>(
            E_NOTIMPL,
            "Not implemented".into(),
        ))
    }

    fn Invoke(&self, result: Option<&IMFAsyncResult>) -> Result<()> {
        unsafe {
            let session = &self.this;
            let session_interface = session.session.as_ref().unwrap();
            let event: IMFMediaEvent = session_interface.EndGetEvent(result)?;

            let event_type = event.GetType()?;
            let status = event.GetStatus()?;

            if status.is_err() {
                return Err(windows::core::Error::from(status));
            }

            if event_type == MESessionEnded.0 as u32 {
                session_interface.Close()?;
            } else if event_type == MESessionClosed.0 as u32 {
                SetEvent(session.wait_event);
            }

            if event_type != MESessionClosed.0 as u32 {
                // Get our callback interface again
                let callback: IMFAsyncCallback = session.cast()?;
                session_interface.BeginGetEvent(&callback, None)?;
            }

            Ok(())
        }
    }
}

// impl IUnknown_Impl for Session {
//     fn QueryInterface(
//         &self,
//         iid: *const GUID,
//         interface: *mut *mut std::ffi::c_void,
//     ) -> Result<()> {
//         unsafe { std::ptr::write(interface, std::ptr::null_mut()) };
//         Ok(())
//     }

//     fn AddRef(&self) -> u32 {
//         self.ref_count.fetch_add(1, Ordering::SeqCst) as u32
//     }

//     fn Release(&self) -> u32 {
//         self.ref_count.fetch_sub(1, Ordering::SeqCst) as u32
//     }
// }
