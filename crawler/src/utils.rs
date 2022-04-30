use mongodb::error::{BulkWriteFailure, ErrorKind, WriteFailure};

/// Returns `Ok(Some(_))` if `result` is `Ok`, `Ok(None)` if `result` represents
/// a duplicate write error, or `Err(err)` otherwise.
pub fn ignore_duplicate_error<T>(
    result: mongodb::error::Result<T>,
) -> mongodb::error::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) => {
            let is_duplicate_error = match &*err.kind {
                ErrorKind::BulkWrite(BulkWriteFailure {
                    write_errors: Some(errors),
                    write_concern_error: None,
                    ..
                }) => errors.iter().all(|x| x.code == DUPLICATE_ERROR_CODE),
                ErrorKind::Write(WriteFailure::WriteError(error)) => {
                    error.code == DUPLICATE_ERROR_CODE
                }
                _ => false,
            };

            if is_duplicate_error {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

/// Returns `Ok(None)` if result is `Ok`, `Ok(Some(_))` if `result` represents
/// a duplicate write error (with `_` being the indices of the duplicated
/// original documents, or an empty vector if the update is not bulk), and
/// `Err(err)` otherwise.
pub fn get_duplicate_indices<T>(
    result: mongodb::error::Result<T>,
) -> mongodb::error::Result<Option<Vec<usize>>> {
    match result {
        Ok(_) => Ok(None),
        Err(err) => match &*err.kind {
            ErrorKind::BulkWrite(BulkWriteFailure {
                write_errors: Some(errors),
                write_concern_error: None,
                ..
            }) if errors.iter().all(|x| x.code == DUPLICATE_ERROR_CODE) => {
                Ok(Some(errors.iter().map(|x| x.index).collect()))
            }
            ErrorKind::Write(WriteFailure::WriteError(error))
                if error.code == DUPLICATE_ERROR_CODE =>
            {
                Ok(Some(vec![]))
            }
            _ => Err(err),
        },
    }
}

const DUPLICATE_ERROR_CODE: i32 = 11_000;
