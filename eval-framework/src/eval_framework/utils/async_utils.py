"""Async utility functions for concurrent operations."""

import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


def to_async(func: Callable[..., R]) -> Callable[..., Coroutine[Any, Any, R]]:
    """Convert a synchronous function to an async function.
    
    Args:
        func: The synchronous function to convert
        
    Returns:
        An async function that runs the original function in a thread pool
    """
    if inspect.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    return wrapper


async def gather_with_concurrency(
    n: int, *tasks: Coroutine[Any, Any, T]
) -> list[T]:
    """Run tasks with a limit on concurrency.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def run_in_threadpool(
    func: Callable[..., R], *args: Any, **kwargs: Any
) -> R:
    """Run a function in a thread pool.
    
    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, *args, **kwargs)
    )


class AsyncThreadPool:
    """Thread pool for running CPU-bound tasks asynchronously."""

    def __init__(self, max_workers: int | None = None):
        """Initialize the thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run(
        self, func: Callable[..., R], *args: Any, **kwargs: Any
    ) -> R:
        """Run a function in the thread pool.
        
        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(func, *args, **kwargs)
        )

    async def map(
        self, func: Callable[[T], R], items: list[T]
    ) -> list[R]:
        """Map a function over items in the thread pool.
        
        Args:
            func: Function to apply
            items: Items to process
            
        Returns:
            List of results
        """
        return await asyncio.gather(
            *(self.run(func, item) for item in items)
        )

    def __del__(self):
        """Clean up the thread pool."""
        self._executor.shutdown(wait=False)


async def retry_async(
    func: Callable[..., Coroutine[Any, Any, R]],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> R:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        **kwargs: Keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception  # type: ignore 