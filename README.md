Data 출처 : https://www.kaggle.com/supergus/multistage-continuousflow-manufacturing-process



# Python 코딩 스타일 가이드

> [Google python style guide](https://google.github.io/styleguide/pyguide.html) 를 참고하였으며
>
> 프로젝트에서 사용하기로 정한 사항입니다.

<br>

* 전역변수 사용 자제

  코딩 혼자하는거 아니니 불가피한 경우에만 사용합시다.

  사용한다면 주석으로 반드시 명시해놓으세요

<br>

* 변수 네이밍

  변수 네이밍 방법은 아래와 같이 정합니다.

  | 타입                 | Public               | Internal                          |
  | -------------------- | -------------------- | --------------------------------- |
  | 패키지               | `lower_with_under`   |                                   |
  | 모듈                 | `lower_with_under`   | `_lower_with_under`               |
  | 클래스               | `CapWords`           | `_CapWords`                       |
  | 예외                 | `CapWords`           |                                   |
  | 함수                 | `lower_with_under()` | `_lower_with_under()`             |
  | 글로벌/클래스 상수   | `CAPS_WITH_UNDER`    | `_CAPS_WITH_UNDER`                |
  | 글로벌/클래스 변수   | `lower_with_under`   | `_lower_with_under`               |
  | 인스턴스 변수        | `lower_with_under`   | `_lower_with_under`               |
  | 메서드 이름          | `lower_with_under()` | `_lower_with_under()` (protected) |
  | 함수/메서드 매개변수 | `lower_with_under`   |                                   |
  | 지역 변수            | `lower_with_under`   |                                   |

<br>

* Docstring

  클래스나 함수에 대한 설명, 인수의 타입, 리턴값의 타입, 인수와 리턴값에 대한 설명이 포함되도록 작성

  * 함수

    ```python
    def fetch_smalltable_rows(table_handle: smalltable.Table,
                            keys: Sequence[Union[bytes, str]],
                            require_all_keys: bool = False,
                          ) -> Mapping[bytes, Tuple[str]]:
      """Fetches rows from a Smalltable.
    
      Retrieves rows pertaining to the given keys from the Table instance
      represented by table_handle.  String keys will be UTF-8 encoded.
    
      Args:
          table_handle: An open smalltable.Table instance.
          keys: A sequence of strings representing the key of each table
            row to fetch.  String keys will be UTF-8 encoded.
          require_all_keys: Optional; If require_all_keys is True only
            rows with values set for all keys will be returned.
    
      Returns:
          A dict mapping keys to the corresponding table row data
          fetched. Each row is represented as a tuple of strings. For
          example:
    
          {b'Serak': ('Rigel VII', 'Preparer'),
          b'Zim': ('Irk', 'Invader'),
          b'Lrrr': ('Omicron Persei 8', 'Emperor')}
    
          Returned keys are always bytes.  If a key from the keys argument is
          missing from the dictionary, then that row was not found in the
          table (and require_all_keys must have been False).
    
      Raises:
          IOError: An error occurred accessing the smalltable.
      """
    ```

  * 클래스

    ```python
    class SampleClass:
        """Summary of class here.
    
        Longer class information....
        Longer class information....
    
        Attributes:
            likes_spam: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """
    
        def __init__(self, likes_spam=False):
            """Inits SampleClass with blah."""
            self.likes_spam = likes_spam
            self.eggs = 0
    
        def public_method(self):
            """Performs operation blah."""
    ```

    

