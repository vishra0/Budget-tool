function deleteTransaction(transactionId) {
  fetch(`/delete_transaction/${transactionId}`, {
    method: 'DELETE',
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      location.reload();
    } else {
      alert('Failed to delete transaction');
    }
  });
}