const infoBtn = document.getElementById('infoBtn');
const walletBtn = document.getElementById('walletBtn');
const infoContent = document.getElementById('infoContent');
const walletContent = document.getElementById('walletContent');

infoBtn.addEventListener('click', function () {
  infoContent.style.display = 'block';
  infoContent.style.opacity = '1';
  walletContent.style.display = 'none';
  walletContent.style.opacity = '0';
})

walletBtn.addEventListener('click', function () {
  walletContent.style.display = 'block';
  walletContent.style.opacity = '1';
  infoContent.style.display = 'none';
  infoContent.style.opacity = '0';
})



// 나의 수도 소비량 차트
var options = {
  series: [{
    name: '소비량',
    data: [2.3, 3.1, 4.0, 10.1, 4.0, 3.6]
  }],
  chart: {
    width: 800,
    height: 400,
    type: 'bar',
    toolbar: {
      show: false,
    },
  },
  plotOptions: {
    bar: {
      borderRadius: 10,
      dataLabels: {
        position: 'top', // top, center, bottom
      },
    }
  },
  dataLabels: {
    enabled: true,
    formatter: function (val) {
      return val + "L";
    },
    offsetY: -20,
    style: {
      fontSize: '15px',
      colors: ["#304758"]
    }
  },
  xaxis: {
    title: {
      text: '2023 💧',
      offsetY: -30,
      style: {
        fontSize: '18px'
      }
    },
    categories: ["1월", "2월", "3월", "4월", "5월", "6월",],
    position: 'top',
    labels: {
      style: {
        fontSize: '20px',
        fontWeight: 'bold',
      }
    },
    axisBorder: {
      show: false
    },
    axisTicks: {
      show: false
    },
    crosshairs: {
      fill: {
        type: 'gradient',
        gradient: {
          colorFrom: '#D8E3F0',
          colorTo: '#BED1E6',
          stops: [0, 100],
          opacityFrom: 0.4,
          opacityTo: 0.5,
        }
      }
    },
    tooltip: {
      enabled: true,
    }
  },
  yaxis: {
    axisBorder: {
      show: false
    },
    axisTicks: {
      show: false,
    },
    labels: {
      show: false,
      formatter: function (val) {
        return val + "L";
      }
    }
  },
  title: {
    text: '2023 💧',
    floating: true,
    offsetY: 470,
    align: 'center',
    style: {
      color: '#444',
      fontSize: '18px'
    },
    margin: 15,
  }
};

var chart = new ApexCharts(document.querySelector("#chart"), options);
chart.render();



