//! “Lifted” from itertools
pub struct UntilDone<'a, I, E: 'a> {
    error: &'a mut Result<(), E>,
    iter: I,
}

impl<'a, I, T, E> Iterator for UntilDone<'a, I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(Ok(x)) => Some(x),
            Some(Err(e)) => {
                *self.error = Err(e);
                None
            }
            None => None,
        }
    }
}

pub trait Until<T> {
    /// “Lift” a function of the values of an iterator so that it can process
    /// an iterator of `Result` values instead.
    fn until_done<E, R>(self, f: impl FnOnce(UntilDone<Self, E>) -> R) -> Result<R, E>;
}

impl<T, I: Iterator<Item = T>> Until<T> for I {
    fn until_done<E, R>(self, f: impl FnOnce(UntilDone<Self, E>) -> R) -> Result<R, E> {
        let mut error = Ok(());

        let result = f(UntilDone {
            error: &mut error,
            iter: self,
        });

        error.map(|_| result)
    }
}
